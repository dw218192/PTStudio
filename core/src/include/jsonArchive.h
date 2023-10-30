#pragma once
#include "nlohmann/json.hpp"
#include "archive.h"

struct JsonArchive : Archive {
	auto operator<<(bool value) -> Archive& override;
	auto operator<<(int value) -> Archive& override;
	auto operator<<(float value) -> Archive& override;
	auto operator<<(double value) -> Archive& override;
	auto operator>>(bool& value) -> Archive& override;
	auto operator<<(std::string_view value) -> Archive& override;
	auto operator>>(int& value) -> Archive& override;
	auto operator>>(float& value) -> Archive& override;
	auto operator>>(double& value) -> Archive& override;
	auto operator>>(std::string& value) -> Archive& override;
	auto str() const -> std::string_view override;

private:
	nlohmann::json m_json;
}; 