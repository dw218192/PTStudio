#include "glResource.h"
#include <tcb/span.hpp>
#include "utils.h"

struct GLBuffer : GLResource<GLBuffer> {
	template<typename T>
	void set_data(tcb::span<T const> data, GLenum usage);
};



template <typename T>
void GLBuffer::set_data(tcb::span<T const> data, GLenum usage) {

}
