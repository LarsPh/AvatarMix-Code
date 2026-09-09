DEBUG_PORTS = {
    'convert_to_neus2': 5678,
    'convert_to_avatarrex': 5679,
    'clean_mesh': 5680,
    'process_mesh': 5681,
    'swapping': 5682,
    'render_swapped_gaussians': 5683,
    'refine_rendered_images': 5684,
    'neus2_train_swapped': 5685,
    'clean_mesh_swapped': 5686,
    'process_mesh_swapped': 5687,
}

ACTORSHQ_GENDER_MAP = {


    "Actor01": "female",
    "Actor02": "male",
    "Actor03": "female",
    "Actor04": "female",
    "Actor05": "male",
    "Actor06": "female",
    "Actor07": "male",
    "Actor08": "male",
}

DATA_TYPES = ['thuman2', 'actorshq', 'avatarrex', 'mvhumannet', 'talkbody4d']

NEUS2_HUMAN_SCALE = {
    "thuman2": 1.0,
    "avatarrex": 0.33,
    "actorshq": 0.5,
    "mvhumannet": 0.5,

    "talkbody4d": 0.55,
}

NEUS2_HUMAN_OFFSET = {
    "thuman2": [0.5, 0.5, 0.5],
    "avatarrex": [0.5, 0.5, 0.5],
    "actorshq": [0.5, 0.0, 0.5],
    "mvhumannet": [0.5, 0.5, 1.0],
    "talkbody4d": [0.5, 0.98, 0.5],
}
