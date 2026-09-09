#include "TensorboardLogger.hpp"

#include <polyfem/utils/Logger.hpp>

#include <filesystem>
#include <chrono>
#include <sstream>

#ifdef POLYFEM_WITH_TENSORBOARD
#include <tensorboard_logger.h>
#endif

namespace polyfem::utils
{
#ifdef POLYFEM_WITH_TENSORBOARD
	struct TensorboardLogger::Impl
	{
		std::unique_ptr<TensorBoardLogger> writer;
	};

	void TensorboardLogger::ImplDeleter::operator()(Impl *p) noexcept
	{
		delete p;
	}
#endif

	TensorboardLogger::~TensorboardLogger() = default;

	static std::string make_event_file(const std::string &dir)
	{
		// TensorBoardLogger requires the basename to contain "tfevents".
		const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
							std::chrono::system_clock::now().time_since_epoch())
							.count();
		std::ostringstream oss;
		oss << "events.out.tfevents." << ts;
		return (std::filesystem::path(dir) / oss.str()).string();
	}

	bool TensorboardLogger::init(const std::string &output_dir, const std::string &log_dir, const Options &opt)
	{
		opt_ = opt;
		enabled_ = false;

		if (!opt_.enabled)
			return false;

#ifndef POLYFEM_WITH_TENSORBOARD
		logger().warn("TensorBoard logging requested but POLYFEM_WITH_TENSORBOARD is OFF; ignoring.");
		return false;
#else
		std::filesystem::path dir = log_dir.empty()
										? (std::filesystem::path(output_dir) / "tb")
										: std::filesystem::path(log_dir);
		std::filesystem::create_directories(dir);

		const std::string event_file = make_event_file(dir.string());

		TensorBoardLoggerOptions tbopt;
		tbopt.max_queue_size(opt_.max_queue_size);
		tbopt.flush_period_s(opt_.flush_period_s);
		tbopt.resume(opt_.resume);

		impl_.reset(new Impl());
		impl_->writer = std::make_unique<TensorBoardLogger>(event_file, tbopt);

		enabled_ = true;
		logger().info("TensorBoard logging enabled at {}", dir.string());
		return true;
#endif
	}

	void TensorboardLogger::add_scalar(const std::string &tag, int64_t step, double value)
	{
#ifdef POLYFEM_WITH_TENSORBOARD
		if (!enabled_ || !impl_ || !impl_->writer)
			return;
		impl_->writer->add_scalar(tag, int(step), value);
#else
		(void)tag;
		(void)step;
		(void)value;
#endif
	}
} // namespace polyfem::utils

