#pragma once
#define NO_COPY_MOVE(Ty)               \
    Ty(Ty const&) = delete;            \
    Ty& operator=(Ty const&) = delete; \
    Ty(Ty&&) = delete;                 \
    Ty& operator=(Ty&&) = delete

#define DEFAULT_COPY_MOVE(Ty)           \
    Ty(Ty const&) = default;            \
    Ty& operator=(Ty const&) = default; \
    Ty(Ty&&) = default;                 \
    Ty& operator=(Ty&&) = default

#define NO_COPY(Ty)         \
    Ty(Ty const&) = delete; \
    Ty& operator=(Ty const&) = delete
