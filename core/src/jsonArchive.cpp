#include "jsonArchive.h"


auto JsonArchive::operator<<(bool value) -> Archive& {
}

auto JsonArchive::operator<<(int value) -> Archive& {
}

auto JsonArchive::operator<<(float value) -> Archive& {
}

auto JsonArchive::operator<<(double value) -> Archive& {
}

auto JsonArchive::operator>>(bool& value) -> Archive& {
}

auto JsonArchive::operator<<(std::string_view value) -> Archive& {
}

auto JsonArchive::operator>>(int& value) -> Archive& {
}

auto JsonArchive::operator>>(float& value) -> Archive& {
}

auto JsonArchive::operator>>(double& value) -> Archive& {
}

auto JsonArchive::operator>>(std::string& value) -> Archive& {
}

auto JsonArchive::str() const -> std::string_view {
	return "";
}
