#include <tcb/span.hpp>
#include <string>

#include "glResource.h"
#include "utils.h"

struct GLBuffer;
using BufferRef = GLResRef<GLBuffer>;

struct GLBuffer final : GLResource {
	static auto create(GLenum target) -> tl::expected<BufferRef, std::string>;

	template<typename T>
	auto set_data(tcb::span<T const> data, GLenum usage = GL_STATIC_DRAW) -> tl::expected<void, std::string>;

private:
	GLBuffer(GLenum target) noexcept;
	~GLBuffer() noexcept override;

	GLuint m_target;
};

template <typename T>
auto GLBuffer::set_data(tcb::span<T const> data, GLenum usage) -> tl::expected<void, std::string> {
	if (data.empty()) {
		return {};
	}

	glBindBuffer(m_target,m_handle);
	CHECK_GL_ERROR();
	glBufferData(m_target, data.size() * sizeof(T), data.data(), usage);
	CHECK_GL_ERROR();
	glBindBuffer(m_target, 0);
	CHECK_GL_ERROR();

	return {};
}
