#include "glFrameBuffer.h"

#include <core/legacy/utils.h>

#include "glRenderBuffer.h"
#include "glTexture.h"

using namespace PTS;

static constexpr GLenum k_color_attachments[] = {
    GL_COLOR_ATTACHMENT0,  GL_COLOR_ATTACHMENT1,  GL_COLOR_ATTACHMENT2,  GL_COLOR_ATTACHMENT3,
    GL_COLOR_ATTACHMENT4,  GL_COLOR_ATTACHMENT5,  GL_COLOR_ATTACHMENT6,  GL_COLOR_ATTACHMENT7,
    GL_COLOR_ATTACHMENT8,  GL_COLOR_ATTACHMENT9,  GL_COLOR_ATTACHMENT10, GL_COLOR_ATTACHMENT11,
    GL_COLOR_ATTACHMENT12, GL_COLOR_ATTACHMENT13, GL_COLOR_ATTACHMENT14, GL_COLOR_ATTACHMENT15,
};

auto GLFrameBuffer::create() -> tl::expected<GLFrameBufferRef, std::string> {
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    CHECK_GL_ERROR();
    return GLFrameBufferRef{new GLFrameBuffer{fbo}, GLResourceDeleter{}};
}

auto GLFrameBuffer::bind() const noexcept -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR("frame buffer is not valid");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    CHECK_GL_ERROR();
    return {};
}

GLFrameBuffer::GLFrameBuffer(GLFrameBuffer&& other) noexcept {
    swap(std::move(other));
}
auto GLFrameBuffer::operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer& {
    swap(std::move(other));
    return *this;
}

void GLFrameBuffer::unbind() noexcept {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

auto GLFrameBuffer::attach(unsigned width, unsigned height,
                           std::initializer_list<AttachmentDesc> descs)
    -> tl::expected<void, std::string> {
    for (auto const& desc : descs) {
        if (desc.attachment == GL_DEPTH_ATTACHMENT) {
            GLRenderBufferRef rbo;
            TL_TRY_ASSIGN(rbo, GLRenderBuffer::create(width, height, desc.format));
            TL_CHECK_AND_PASS(rbo->bind());
            {
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, desc.attachment, GL_RENDERBUFFER,
                                          rbo->handle());
                CHECK_GL_ERROR();
            }
            rbo->unbind();

            m_rbo_attchs[desc.attachment] = std::move(rbo);

        } else if (std::find(std::begin(k_color_attachments), std::end(k_color_attachments),
                             desc.attachment) != std::end(k_color_attachments)) {
            GLTextureRef tex;
            TL_TRY_ASSIGN(tex, GLTexture::create(width, height, desc.format, desc.params));
            TL_CHECK_AND_PASS(tex->bind());
            {
                glFramebufferTexture(GL_FRAMEBUFFER, desc.attachment, tex->handle(), 0);
                CHECK_GL_ERROR();
            }
            tex->unbind();

            m_tex_attchs[desc.attachment] = std::move(tex);

        } else {
            return TL_ERROR("unsupported attachment");
        }
    }
    return {};
}

auto GLFrameBuffer::set_draw_buffer(GLenum attachment) const -> tl::expected<void, std::string> {
    glDrawBuffers(1, &attachment);
    CHECK_GL_ERROR();
    return {};
}

auto GLFrameBuffer::resize(unsigned width,
                           unsigned height) noexcept -> tl::expected<void, std::string> {
    for (auto&& [attachment, rbo] : m_rbo_attchs) {
        TL_CHECK_AND_PASS(rbo->bind());
        TL_CHECK_AND_PASS(rbo->resize(width, height));
        rbo->unbind();
    }
    for (auto&& [attachment, tex] : m_tex_attchs) {
        TL_CHECK_AND_PASS(tex->bind());
        TL_CHECK_AND_PASS(tex->resize(width, height));
        tex->unbind();
    }
    return {};
}

auto GLFrameBuffer::get_texture(GLenum attachment) const noexcept -> GLTexture const* {
    if (auto const it = m_tex_attchs.find(attachment); it != m_tex_attchs.end()) {
        return it->second.get();
    }
    return nullptr;
}

auto GLFrameBuffer::get_render_buffer(GLenum attachment) const noexcept -> GLRenderBuffer const* {
    if (auto const it = m_rbo_attchs.find(attachment); it != m_rbo_attchs.end()) {
        return it->second.get();
    }
    return nullptr;
}

auto GLFrameBuffer::swap_render_buffer(GLenum attachment, GLRenderBufferRef buf) noexcept
    -> tl::expected<GLRenderBufferRef, std::string> {
    if (auto const it = m_rbo_attchs.find(attachment); it != m_rbo_attchs.end()) {
        auto old = std::move(it->second);

        if (buf) {
            TL_CHECK(buf->bind());
            {
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER,
                                          buf->handle());
                CHECK_GL_ERROR();
            }
            buf->unbind();

            it->second = std::move(buf);
        } else {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, 0);
            CHECK_GL_ERROR();

            m_rbo_attchs.erase(it);
        }

        return old;
    }
    return TL_ERROR("attachment not found");
}

auto GLFrameBuffer::swap_texture(GLenum attachment, GLTextureRef tex) noexcept
    -> tl::expected<GLTextureRef, std::string> {
    if (auto const it = m_tex_attchs.find(attachment); it != m_tex_attchs.end()) {
        auto old = std::move(it->second);

        if (tex) {
            TL_CHECK(tex->bind());
            {
                glFramebufferTexture(GL_FRAMEBUFFER, attachment, tex->handle(), 0);
                CHECK_GL_ERROR();
            }
            tex->unbind();
            it->second = std::move(tex);
        } else {
            glFramebufferTexture(GL_FRAMEBUFFER, attachment, 0, 0);
            CHECK_GL_ERROR();

            m_tex_attchs.erase(it);
        }

        return old;
    }
    return TL_ERROR("attachment not found");
}

auto GLFrameBuffer::swap_texture(GLenum attachment1,
                                 GLenum attachment2) noexcept -> tl::expected<void, std::string> {
    if (auto const it1 = m_tex_attchs.find(attachment1); it1 != m_tex_attchs.end()) {
        if (auto const it2 = m_tex_attchs.find(attachment2); it2 != m_tex_attchs.end()) {
            std::swap(it1->second, it2->second);
            return {};
        }
        return TL_ERROR("attachment not found");
    }
    return TL_ERROR("attachment not found");
}

auto GLFrameBuffer::clear(glm::vec3 color,
                          float depth) const noexcept -> tl::expected<void, std::string> {
    glClearColor(color.r, color.g, color.b, 1.0f);
    CHECK_GL_ERROR();

    glClearDepth(depth);
    CHECK_GL_ERROR();

    glClear(GL_DEPTH_BUFFER_BIT);
    CHECK_GL_ERROR();

    // save the draw buffer
    GLint draw_buffer;
    glGetIntegerv(GL_DRAW_BUFFER, &draw_buffer);
    CHECK_GL_ERROR();

    for (auto&& [attachment, tex] : m_tex_attchs) {
        TL_CHECK_AND_PASS(set_draw_buffer(attachment));

        glClear(GL_COLOR_BUFFER_BIT);
        CHECK_GL_ERROR();
    }

    // restore the draw buffer
    TL_CHECK_AND_PASS(set_draw_buffer(draw_buffer));

    return {};
}

auto GLFrameBuffer::clear_color(GLenum attachment,
                                glm::vec3 color) const noexcept -> tl::expected<void, std::string> {
    // save the draw buffer
    GLint draw_buffer;
    glGetIntegerv(GL_DRAW_BUFFER, &draw_buffer);
    CHECK_GL_ERROR();

    TL_CHECK_AND_PASS(set_draw_buffer(attachment));

    glClearColor(color.r, color.g, color.b, 1.0f);
    CHECK_GL_ERROR();

    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_GL_ERROR();

    // restore the draw buffer
    TL_CHECK_AND_PASS(set_draw_buffer(draw_buffer));

    return {};
}

void GLFrameBuffer::swap(GLFrameBuffer&& other) noexcept {
    if (this == &other) {
        return;
    }
    m_rbo_attchs.swap(other.m_rbo_attchs);
    m_tex_attchs.swap(other.m_tex_attchs);
    this->GLResource::swap(std::move(other));
}

GLFrameBuffer::GLFrameBuffer(GLuint handle) noexcept : GLResource(handle) {
}

GLFrameBuffer::~GLFrameBuffer() noexcept {
    if (m_handle) {
        glDeleteFramebuffers(1, &m_handle);
    }
}