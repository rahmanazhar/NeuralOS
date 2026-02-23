# ── Compiler Warning Configuration ───────────────────────────────────────────
#
# Provides nos_set_warnings(target) to enable strict compiler warnings.

function(nos_set_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wconversion
            -Wnon-virtual-dtor
        >
    )
endfunction()
