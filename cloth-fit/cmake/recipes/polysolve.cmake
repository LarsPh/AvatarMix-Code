# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

option(POLYSOLVE_WITH_CHOLMOD       "Enable Cholmod library"                             ON)
option(POLYSOLVE_WITH_UMFPACK       "Enable UmfPack library"                            OFF)
option(POLYSOLVE_WITH_SUPERLU       "Enable SuperLU library"                            OFF)
option(POLYSOLVE_WITH_CUSOLVER      "Enable cuSOLVER library"                           OFF)
option(POLYSOLVE_WITH_PARDISO       "Enable Pardiso library"                            OFF)
option(POLYSOLVE_WITH_HYPRE         "Enable hypre"                                      OFF)
option(POLYSOLVE_WITH_AMGCL         "Use AMGCL"                                         OFF)
option(POLYSOLVE_WITH_SPECTRA       "Enable Spectra library"                            OFF)

include(CPM)
CPMAddPackage("gh:Huangzizhou/polysolve#ee553dc6a330ef70bcc6ba2f05653fecd201c05f")
