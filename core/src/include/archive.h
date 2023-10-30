#pragma once

#include <tl/expected.hpp>
#include <string_view>
#include <string>

/**
 * \brief Interface for any serializer that can be used to serialize/deserialize objects\n
 * serializers should implement operator<< and operator>> for simple types\n
 * all complex serializable types should implement T& operator<<(Archive&, T) and T& operator>>(Archive&, T)
*/
struct Archive {
    virtual ~Archive() = default;
    virtual auto operator<<(bool value) -> Archive& = 0;
    virtual auto operator<<(int value) -> Archive& = 0;
    virtual auto operator<<(float value) -> Archive& = 0;
    virtual auto operator<<(double value) -> Archive& = 0;
    virtual auto operator<<(std::string_view value) -> Archive& = 0;
    virtual auto operator>>(bool& value) -> Archive& = 0;
    virtual auto operator>>(int& value) -> Archive& = 0;
    virtual auto operator>>(float& value) -> Archive& = 0;
    virtual auto operator>>(double& value) -> Archive& = 0;
    virtual auto operator>>(std::string& value) -> Archive& = 0;

	virtual auto str() const -> std::string_view = 0;
};