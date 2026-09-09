#pragma once

#include <string>
#include <memory>

namespace polyfem::utils
{
	/// Minimal wrapper around a TensorBoard event writer.
	/// When POLYFEM_WITH_TENSORBOARD is not enabled, this becomes a no-op.
	class TensorboardLogger
	{
	public:
		struct Options
		{
			bool enabled = false;
			// Defaults are intentionally conservative to reduce runtime overhead.
			int log_every = 10;        // log scalars every N post_step calls
			int energy_every = 50;     // compute energy every N (can be expensive)
			size_t flush_period_s = 3; // library flush thread period
			size_t max_queue_size = 100000;
			bool resume = false;
		};

		TensorboardLogger() = default;
		~TensorboardLogger();

		bool is_enabled() const { return enabled_; }

		/// Initialize the writer under <output_dir>/tb by default.
		/// If log_dir is empty, defaults to <output_dir>/tb.
		bool init(const std::string &output_dir, const std::string &log_dir, const Options &opt);

		void add_scalar(const std::string &tag, int64_t step, double value);

		const Options &options() const { return opt_; }

	private:
		bool enabled_ = false;
		Options opt_;

#ifdef POLYFEM_WITH_TENSORBOARD
		struct Impl;
		struct ImplDeleter
		{
			void operator()(Impl *p) noexcept;
		};
		std::unique_ptr<Impl, ImplDeleter> impl_;
#endif
	};
} // namespace polyfem::utils

