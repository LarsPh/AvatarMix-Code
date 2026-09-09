# Cloth Retargeting Extensions for Human Meshes

This document summarizes the modifications we made on top of the original `cloth-fit` codebase to better handle human meshes (with holes, floaters, and fragile regions like hands) and to preserve 3D Gaussian structure.

---

## 1. Skeleton Pre-pass Optimization (Inside SDF + Rigidity)

### 1.1. Goal

We first adjust the **source skeleton** so that all bones lie **inside** the human (garment) mesh, while changing the skeleton as rigidly as possible. This is a pre-pass that runs before avatar projection and garment optimization.

Let the skeleton vertices be
\[
X = \{x_k\}_{k=1}^{N_s}, \quad x_k \in \mathbb{R}^3
\]
and skeleton edges (bones) be \(E\).

We build a signed distance field (SDF) \(\phi(\cdot)\) of the human mesh using OpenVDB, with several robustness tricks:

- **Flood-fill sign**: `flood_fill_sign = true`
- **Hole closing in SDF**: `close_holes_voxels > 0` (dilate then erode)
- **Capping open boundaries**: `cap_open_boundaries = true` (fan triangulation of open loops before SDF)

### 1.2. SDF Inside Penalty

For each bone \(e = (i, j) \in E\), we sample points along the segment:
\[
p_e(t) = (1 - t)x_i + t x_j, \quad t \in [0, 1],
\]
with a small set of values \(t_s\). For each sample \(p_s = p_e(t_s)\), we evaluate the SDF:
\[
\phi_s = \phi(p_s),
\]
and define an “inside” penalty (with margin \(\delta \ge 0\)):
\[
E_{\text{inside}}(X) = \sum_s \max\bigl(0,\ \phi_s + \delta\bigr)^2.
\]

### 1.3. Bone Length Regularization

We preserve the original bone lengths. Let
\[
L_{ij}^2 = \|x_i^{(0)} - x_j^{(0)}\|^2
\]
be the squared rest length for edge \((i, j) \in E\). Then:
\[
E_{\text{length}}(X) = \sum_{(i,j)\in E} \left( \|x_i - x_j\|^2 - L_{ij}^2 \right)^2.
\]

### 1.4. Root Anchor

Optionally, we softly anchor a root joint \(r\) to its initial position:
\[
E_{\text{anchor}}(X) = \|x_r - x_r^{(0)}\|^2.
\]

### 1.5. Pre-pass Objective

The pre-pass objective is:
\[
E_{\text{pre}}(X) = w_{\text{inside}} E_{\text{inside}}(X)
+ w_{\text{len}} E_{\text{length}}(X)
+ w_{\text{anch}} E_{\text{anchor}}(X).
\]

We solve
\[
\min_X E_{\text{pre}}(X)
\]
with a nonlinear solver, starting from the original skeleton.

---

## 2. Main Garment Retargeting with Semantic Masks

We apply the original retargeting pipeline (avatar projection to skeleton, exploding, etc.), but augment it with **per-vertex masks** that let us treat semantic regions (skin, hands, arms, etc.) differently.

### 2.1. Variables

- Garment vertices: \(V_g = \{v_i\} \in \mathbb{R}^{N_g \times 3}\)
- “Skinny” avatar vertices: \(X_{\text{skinny}}\)
- Exploded avatar vertices: \(X_{\text{nc}}\)

We use an incremental parameter \(\alpha \in [0,1]\) with `incremental_steps` substeps:
\[
X_\alpha = X_{\text{skinny}} + \alpha (X_{\text{nc}} - X_{\text{skinny}}).
\]

### 2.2. Similarity (ARAP-like) Regularization

SimilarityForm encourages neighboring triangles to remain locally similar (an ARAP-like shape preservation term). In simplified form:
\[
E_{\text{sim}}(V_g) \approx \sum_{t} A_t \,\|F_t(V_g) - R_t\|^2,
\]
where \(A_t\) is a local area weight, \(F_t\) is a local deformation descriptor, and \(R_t\) are “idealized” similar configurations.

We introduce per-vertex multipliers \(m_{\text{sim}}(i)\) via `similarity_weight_masks`. For each similarity stencil involving vertex set \(\mathcal{I}\), we use the average multiplier:
\[
\bar{m}_{\text{sim}} = \frac{1}{|\mathcal{I}|} \sum_{i\in\mathcal{I}} m_{\text{sim}}(i).
\]
Then:
\[
E_{\text{sim}}(V_g) = \sum_{t} \bar{m}_{\text{sim}}(t) \, A_t \,\|F_t(V_g) - R_t\|^2.
\]

- High \(m_{\text{sim}}\) on hands/arms: preserves local hand/arm shape (near-rigid).

### 2.3. SDF Fit to Avatar

FitForm pulls the garment toward the true avatar surface (not the skinny). Let \(\phi_A(\cdot)\) be the avatar SDF. We sample each garment face at local sample points \(p\) and penalize positive distances:
\[
E_{\text{fit}}(V_g) = \sum_{p} m_{\text{fit}}(v(p)) \,\psi\bigl(\phi_A(p)\bigr),
\]
where \(\psi(\phi) = \max(0,\phi)^p\) (with some power \(p\)), and \(m_{\text{fit}}\) is a per-vertex weight (from `fit_weight_masks`).

- Large \(m_{\text{fit}}\): tight fit to the avatar (shrinks or expands depending on sign of \(\phi_A\)).
- For hands: we typically use **moderate** \(m_{\text{fit}}\) plus strong Similarity, so the hand shrinks to the SMPL shape but preserves local structure.

### 2.4. Continuation (Avatar Displacement)

The avatar is moved from skinny to target via continuation. At substep \(\alpha\):
\[
d(v) = X_\alpha(v) - X_{\text{skinny}}(v).
\]

We use an augmented Lagrangian to enforce a displacement-BC:
\[
E_{\text{AL}}(X_g) = \sum_{v} \|u(v) - d(v)\|^2,
\]
implemented via PointPenalty + PointLagrangian forms.

We introduce per-avatar-vertex multipliers \(m_{\text{cont}}(v)\) (from `continuation_weight_masks`) and scale the target:
\[
d'(v) = m_{\text{cont}}(v)\, d(v).
\]

- Low \(m_{\text{cont}}\) on hands/arms/neck: damp the continuation pull, preventing “glove-like” inflation driven by the skinny avatar.

> **Important:** In practice, we found these continuation masks helpful but not perfectly behaved in all cases (e.g., slow overall dilation when masks are too aggressive or mapping is imperfect). We treat them as a heuristically useful, but imperfect, control.

### 2.5. Contact Barrier (Optional)

We use IPC-style contact with a barrier potential between avatar and garment only:
\[
E_{\text{contact}}(V_g, X) = \sum_{\text{contact pairs}} \phi_{\text{barrier}}(\text{distance} - d_{\text{hat}}),
\]
activated by `contact.enabled` and parameterized by `contact.dhat` and `solver.contact.barrier_stiffness`.

For difficult self-intersecting meshes, we often **disable contact during the continuation stage** and optionally re-enable it only for the final fit stage.

### 2.6. Total Energy Per Substep

For a given \(\alpha\), the main solve minimizes:
\[
E(V_g) = w_{\text{sim}} E_{\text{sim}}(V_g)
+ w_{\text{fit}} E_{\text{fit}}(V_g)
+ E_{\text{curves}}(V_g)
+ E_{\text{contact}}(V_g, X_\alpha)
+ E_{\text{AL}}(X_\alpha),
\]
with `incremental_steps` to gradually ramp \(\alpha\) from 0 to 1.

---

## 3. Per-vertex Masks and Semantics

We use three mask families:

- **Fit masks** \(m_{\text{fit}}(i)\) (garment vertices)
  - `fit_weight_masks`: per-garment-vertex multipliers.
  - Used to weaken or strengthen SDF fitting in regions like bare skin (hands/arms).

- **Similarity masks** \(m_{\text{sim}}(i)\) (garment region, implemented over collision vertices but only garment faces contribute)
  - `similarity_weight_masks`: per-vertex multipliers for SimilarityForm.
  - Used to increase rigidity (e.g., hands/arms) or soften other regions.

- **Continuation masks** \(m_{\text{cont}}(v)\) (avatar vertices)
  - `continuation_weight_masks`: per-avatar (or garment→avatar mapped) multipliers for the continuation displacement.
  - Used to damp the skinny avatar pull in sensitive regions (hands, neck, etc.).
  - In practice, these masks help, but their effect can be global (through ARAP coupling and AL stiffness), and we observed that overly aggressive masks can slow down dilation; we treat them carefully and report them as a heuristic rather than a perfect solution.

We also support mapping garment masks to avatar space (e.g., “mesh: garment” entries mapped to nearest avatar vertices) and PLY visualizations to inspect which vertices are masked.

---

## 4. Avatar Vertex Removal (Hands/Feet/Head)

To avoid pathological behavior in regions we don’t care to reconstruct (e.g., head/neck/hands/feet for Gaussian fitting), we remove those regions from the avatar before normalization and projection.

### 4.1. Vertex Removal

Given a set of avatar vertex indices \(\mathcal{R}\), we:

1. Mark vertices in \(\mathcal{R}\) as removed.
2. Build a new vertex array \(X'\) consisting of the kept vertices.
3. Keep only faces whose vertices are all kept, remapping them to the new indices.

### 4.2. Skin Weights Update

Given skin weights \(W \in \mathbb{R}^{B \times N}\) (bones \(\times\) avatar verts), we slice columns corresponding to kept vertices:
\[
W' = \begin{bmatrix} \cdots W_{:, v} \cdots \end{bmatrix}_{v \notin \mathcal{R}}.
\]

### 4.3. Translation Preservation

Originally, the avatar centroid used in normalization was:
\[
c_0 = \frac{1}{N}\sum_{v} X(v).
\]

After removal, naively recomputing the centroid from \(X'\) would change the global translation. To preserve translation, we **freeze** the original centroid before removal and use it in the target-side alignment:

- Source side: center and scale the source skeleton.
- Target side:
  - Let \(c_s = \text{mean}(X_{\text{source-skel}})\).
  - Let \(c_A\) be the **original** avatar centroid (before removal).
  - Let \(s_t\) be the target scaling factor.
  - Then the translation is:
    \[
    t = c_s - s_t \, c_A,
    \]
  - and we apply \(x \mapsto s_t x + t\) to the avatar and target skeleton.

This keeps the avatar’s global translation consistent while removing unwanted parts.

---

## 5. Observed Behavior and Limitations

- **Skeleton pre-pass**:
  - Works well when combined with SDF robustness (flood-fill, hole closing, capping).
  - Crucial for meshes with ankle/hem holes or floaters; otherwise initial bone–mesh intersections are common.

- **Similarity + Fit**:
  - High Similarity + moderate Fit on hands/arms can shrink them toward SMPL while preserving shape (avoids “destroying” Gaussians).
  - Fit alone can over-shrink or expand if \(\phi_A\) is noisy; Similarity stabilizes local shape.

- **Continuation masks**:
  - In theory, damping continuation only on hands/arms should localize its effect.
  - In practice, due to ARAP coupling, AL stiffness, and mapping ambiguities, aggressive masks can slow inflation globally and behave less predictably.
  - Treat these masks as a heuristic rather than precise local control.

- **Avatar removal**:
  - Removing head/hands/feet from the avatar (plus centroid freezing) is robust and often simpler: continuation and fit no longer “see” those regions, and we avoid glove/broken-structure issues there altogether.
  - Particularly effective in our human + GS setting, where hands/head often hurt more than they help.

---

## 6. Extension summary

- We introduce a skeleton pre-pass that optimizes bone positions inside a robust SDF of the human mesh, balancing an inside penalty and bone-length regularization.
- We apply semantic per-vertex masks to the similarity, SDF fit, and continuation terms, to preserve hand/arm shape and prevent “glove-like” inflation.
- We optionally remove avatar vertices in unimportant or problematic regions (e.g., hands/head/feet), updating skin weights and preserving global translation via a frozen centroid.
- These extensions allow the original garment retargeting system to reshape human meshes while better preserving fine-scale structures (such as 3D Gaussian distributions) in challenging regions.
