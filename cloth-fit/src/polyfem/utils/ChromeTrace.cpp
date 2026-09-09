#include "ChromeTrace.hpp"

#include <polyfem/utils/Logger.hpp>

#include <chrono>
#include <filesystem>
#include <sstream>

namespace polyfem::utils
{
	ChromeTrace &ChromeTrace::instance()
	{
		static ChromeTrace inst;
		return inst;
	}

	ChromeTrace::~ChromeTrace()
	{
		shutdown();
	}

	bool ChromeTrace::init(const std::string &path)
	{
		std::lock_guard<std::mutex> lock(mtx_);
		if (enabled_)
			return true;
		if (path.empty())
			return false;

		std::error_code ec;
		std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);

		out_.open(path, std::ios::out | std::ios::trunc);
		if (!out_.is_open())
		{
			logger().warn("Failed to open chrome trace file {}", path);
			return false;
		}

		t0_ = std::chrono::steady_clock::now();
		first_event_ = true;
		enabled_ = true;

		out_ << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n";
		out_.flush();
		logger().info("Chrome trace profiling enabled at {}", path);
		return true;
	}

	void ChromeTrace::shutdown()
	{
		std::lock_guard<std::mutex> lock(mtx_);
		if (!enabled_)
			return;
		out_ << "\n]}\n";
		out_.flush();
		out_.close();
		enabled_ = false;
		t0_.reset();
	}

	int64_t ChromeTrace::now_us() const
	{
		if (!t0_)
			return 0;
		const auto dt = std::chrono::steady_clock::now() - *t0_;
		return std::chrono::duration_cast<std::chrono::microseconds>(dt).count();
	}

	void ChromeTrace::write_event_json_locked(const std::string &json)
	{
		if (!out_.is_open())
			return;
		if (!first_event_)
			out_ << ",\n";
		first_event_ = false;
		out_ << json;
	}

	void ChromeTrace::write_complete_event(
		const std::string &name,
		int64_t ts_us,
		int64_t dur_us,
		uint32_t pid,
		uint64_t tid,
		const std::string &cat)
	{
		if (!enabled_)
			return;

		std::ostringstream oss;
		oss << "{"
			<< "\"name\":\"" << name << "\","
			<< "\"cat\":\"" << cat << "\","
			<< "\"ph\":\"X\","
			<< "\"ts\":" << ts_us << ","
			<< "\"dur\":" << dur_us << ","
			<< "\"pid\":" << pid << ","
			<< "\"tid\":" << tid
			<< "}";

		std::lock_guard<std::mutex> lock(mtx_);
		write_event_json_locked(oss.str());
	}

	static uint64_t thread_id_u64()
	{
		return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
	}

	ChromeTraceScope::ChromeTraceScope(const std::string &name, const std::string &cat)
		: name_(name), cat_(cat)
	{
		auto &tr = ChromeTrace::instance();
		if (!tr.enabled())
			return;
		active_ = true;
		start_us_ = tr.now_us();
	}

	ChromeTraceScope::~ChromeTraceScope()
	{
		if (!active_)
			return;
		auto &tr = ChromeTrace::instance();
		const int64_t end_us = tr.now_us();
		tr.write_complete_event(name_, start_us_, std::max<int64_t>(0, end_us - start_us_), 0, thread_id_u64(), cat_);
	}
} // namespace polyfem::utils

