
set(proj OpenSSL)

# Set dependency list
set(${proj}_DEPENDENCIES "")
if(UNIX)
  set(${proj}_DEPENDENCIES zlib)
endif()

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  unset(OPENSSL_INCLUDE_DIR CACHE)
  if(UNIX)
    unset(OPENSSL_CRYPTO_LIBRARY CACHE)
    unset(OPENSSL_SSL_LIBRARY CACHE)
  elseif(WIN32)
    unset(LIB_EAY_DEBUG CACHE)
    unset(LIB_EAY_RELEASE CACHE)
    unset(SSL_EAY_DEBUG CACHE)
    unset(SSL_EAY_RELEASE CACHE)
  endif()
  find_package(OpenSSL REQUIRED)
endif()


if(UNIX)
  set(_has_openssl_libraries
    DEFINED OPENSSL_CRYPTO_LIBRARY
    AND DEFINED OPENSSL_SSL_LIBRARY
    )
elseif(WIN32)
  set(_has_openssl_libraries
    (DEFINED LIB_EAY_DEBUG
    AND DEFINED SSL_EAY_DEBUG)
    OR
    (DEFINED LIB_EAY_RELEASE
    AND DEFINED SSL_EAY_RELEASE)
    )
endif()

if(NOT DEFINED OPENSSL_LIBRARIES
   AND NOT (${_has_openssl_libraries}) AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})

  #------------------------------------------------------------------------------
  if(UNIX)

    set(OPENSSL_DOWNLOAD_VERSION "1.0.2n" CACHE STRING "Version of OpenSSL source package to download")
    set_property(CACHE OPENSSL_DOWNLOAD_VERSION PROPERTY STRINGS "1.0.1e" "1.0.1l" "1.0.2n")

    set(OpenSSL_1.0.1e_URL http://packages.kitware.com/download/item/6172/openssl-1.0.1e.tar.gz)
    set(OpenSSL_1.0.1e_MD5 66bf6f10f060d561929de96f9dfe5b8c)

    set(OpenSSL_1.0.1l_URL http://packages.kitware.com/download/item/7669/openssl-1.0.1l.tar.gz)
    set(OpenSSL_1.0.1l_MD5 cdb22925fc9bc97ccbf1e007661f2aa6)

    set(OpenSSL_1.0.2n_URL https://packages.kitware.com/download/item/10337/openssl-1.0.2n.tar.gz)
    set(OpenSSL_1.0.2n_MD5 13bdc1b1d1ff39b6fd42a255e74676a4)

    if(NOT DEFINED OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_URL)
      message(FATAL_ERROR "There is no source version of OpenSSL ${OPENSSL_DOWNLOAD_VERSION} available.
You could either:
 (1) disable SSL support configuring with option Slicer_USE_PYTHONQT_WITH_OPENSSL:BOOL=OFF
 or
 (2) configure Slicer providing your own version of OpenSSL that matches the version
     of OpenSSL also used to compile Qt library.
     The options to specify are OPENSSL_INCLUDE_DIR, OPENSSL_SSL_LIBRARY, OPENSSL_CRYPTO_LIBRARY.")
    endif()

    #------------------------------------------------------------------------------
    set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})

    include(ExternalProjectForNonCMakeProject)

    # environment
    set(_env_script ${CMAKE_BINARY_DIR}/${proj}_Env.cmake)
    ExternalProject_Write_SetBuildEnv_Commands(${_env_script})
    file(APPEND ${_env_script}
"#------------------------------------------------------------------------------
# Added by '${CMAKE_CURRENT_LIST_FILE}'
include(\"${Slicer_CMAKE_DIR}/ExternalProjectForNonCMakeProject.cmake\")
set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")
set(ENV{VS_UNICODE_OUTPUT} \"\")
")
    if(APPLE)
      file(APPEND ${_env_script}
"# Hint OpenSSL that we prefer a 64-bit build.
  set(ENV{KERNEL_BITS} \"64\")
  # Allow 'sysctl' executable used in OpenSSL config script to be found.
  # This is required when building Slicer using a cronjob where the
  # default environement is restricted.
  set(ENV{PATH} \"/usr/sbin:\$ENV{PATH}\")
")
    endif()

    get_filename_component(_zlib_library_dir ${ZLIB_LIBRARY} PATH)

    # configure step
    set(_configure_script ${CMAKE_BINARY_DIR}/${proj}_configure_step.cmake)
    file(WRITE ${_configure_script}
"include(\"${_env_script}\")
set(${proj}_WORKING_DIR \"${EP_SOURCE_DIR}\")
ExternalProject_Execute(${proj} \"configure-zlib\" cp ${ZLIB_LIBRARY} ${_zlib_library_dir}/libz.a
  )
ExternalProject_Execute(${proj} \"configure\" sh config --with-zlib-lib=${_zlib_library_dir} --with-zlib-include=${ZLIB_INCLUDE_DIR} threads zlib shared
  )
")

    #------------------------------------------------------------------------------
    ExternalProject_Add(${proj}
      ${${proj}_EP_ARGS}
      URL ${OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_URL}
      URL_MD5 ${OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_MD5}
      DOWNLOAD_DIR ${CMAKE_BINARY_DIR}
      SOURCE_DIR ${EP_SOURCE_DIR}
      BUILD_IN_SOURCE 1
      PATCH_COMMAND ${CMAKE_COMMAND} -P ${_configure_script}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make -j1 build_libs
      INSTALL_COMMAND ""
      DEPENDS
        ${${proj}_DEPENDENCIES}
      )

    if(APPLE)
      ExternalProject_Add_Step(${proj} fix_rpath
        COMMAND install_name_tool -id ${EP_SOURCE_DIR}/libcrypto.dylib ${EP_SOURCE_DIR}/libcrypto.dylib
        COMMAND install_name_tool
          -change /usr/local/ssl/lib/libcrypto.1.0.0.dylib ${EP_SOURCE_DIR}/libcrypto.dylib
          -id ${EP_SOURCE_DIR}/libssl.dylib ${EP_SOURCE_DIR}/libssl.dylib
        DEPENDEES build
        )
    endif()

    set(OpenSSL_DIR ${EP_SOURCE_DIR})
    set(OPENSSL_INCLUDE_DIR ${OpenSSL_DIR}/include)
    set(OPENSSL_LIBRARY_DIR ${OpenSSL_DIR})
    set(OPENSSL_EXPORT_LIBRARY_DIR ${OpenSSL_DIR})

    set(OPENSSL_CRYPTO_LIBRARY ${OPENSSL_LIBRARY_DIR}/libcrypto${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(OPENSSL_SSL_LIBRARY ${OPENSSL_LIBRARY_DIR}/libssl${CMAKE_SHARED_LIBRARY_SUFFIX})

    ExternalProject_Message(${proj} "OPENSSL_CRYPTO_LIBRARY:${OPENSSL_CRYPTO_LIBRARY}")
    ExternalProject_Message(${proj} "OPENSSL_SSL_LIBRARY:${OPENSSL_SSL_LIBRARY}")

  #------------------------------------------------------------------------------
  elseif(WIN32)

    set(OPENSSL_DOWNLOAD_VERSION "1.0.1h" CACHE STRING "Version of OpenSSL pre-compiled package ot download.")
    set_property(CACHE OPENSSL_DOWNLOAD_VERSION PROPERTY STRINGS "1.0.1h" "1.0.1l")

    set(_qt_version "${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH}")

    # Starting with Qt 4.8.6, we compiled [1] OpenSSL binaries specifically for each
    # version of Microsoft Visual Studio. To understand the motivation, read below.
    #
    # As explained in [2][3], if a library meets these two conditions:
    #   (a) exposes a pure C interface
    #   (b) is linked against static version of the CRT
    # the version of the compiler and runtime used to build the library
    # shouldn't matter.
    #
    # That said, since OpenSSL is built as a shared library using the '/MD' [4]
    # flag, it is not possible to use the same library with different runtime
    # libraries.
    #
    # We found out that '/MD' was used by inspecting the the file 'ms/ntdll.mak'
    # generated atfer configuring OpenSSL.
    #
    # If you find mistake in this explanation, do not hesitate to submit a patch
    # to fix this text. Thanks.
    #
    # [1] Script used to compile OpenSSL: https://gist.github.com/jcfr/6030240
    # [2] http://siomsystems.com/mixing-visual-studio-versions/
    # [3] http://bytes.com/topic/net/answers/505515-compile-different-versions-visual-studio
    # [4] http://msdn.microsoft.com/en-us/library/2kzt1wy3.aspx

    set(_error_msg "There is no pre-compiled version of OpenSSL ${OPENSSL_DOWNLOAD_VERSION} available for
this version of visual studio [${MSVC_VERSION}]. You could either:
 (1) disable SSL support configuring with option Slicer_USE_PYTHONQT_WITH_OPENSSL:BOOL=OFF
 or
 (2) configure Slicer providing your own version of OpenSSL that matches the version
     of OpenSSL also used to compile Qt library.
     The options to specify are OPENSSL_INCLUDE_DIR, LIB_EAY_DEBUG, LIB_EAY_RELEASE,
     SSL_EAY_DEBUG and SSL_EAY_RELEASE.")

    #--------------------
    if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32-bit

      # OpenSSL 1.0.1h
      # VS2008
      set(OpenSSL_1.0.1h_1500_URL http://packages.kitware.com/download/item/6093/OpenSSL_1_0_1h-install-msvc1500-32.tar.gz)
      set(OpenSSL_1.0.1h_1500_MD5 8b110bb48063223c3b9f3a99f1fa9067)
      # VS2010
      set(OpenSSL_1.0.1h_1600_URL http://packages.kitware.com/download/item/6096/OpenSSL_1_0_1h-install-msvc1600-32.tar.gz)
      set(OpenSSL_1.0.1h_1600_MD5 e80269ae7969276977a342cccc1df5c5)
      # VS2013
      set(OpenSSL_1.0.1h_1800_URL http://packages.kitware.com/download/bitstream/8918/OpenSSL_1_0_1h-install-msvc1800-32.tar.gz)
      set(OpenSSL_1.0.1h_1800_MD5 f10ceb422ab37f2b0bd5e225c74fd1d4)
      # VS2015
      set(OpenSSL_1.0.1h_1900_URL https://packages.kitware.com/download/item/10334/OpenSSL_1_0_1h-install-msvc1900-32.tar.gz)
      set(OpenSSL_1.0.1h_1900_MD5 e0e26ae6ac5693d266c804e738d7aa14)

      # OpenSSL 1.0.1l
      # VS2008
      set(OpenSSL_1.0.1l_1500_URL http://packages.kitware.com/download/item/7689/OpenSSL_1_0_1l-install-msvc1500-32.tar.gz)
      set(OpenSSL_1.0.1l_1500_MD5 85218e0ea7bd15457f66cc04170560d5)
      # VS2010
      set(OpenSSL_1.0.1l_1600_URL http://packages.kitware.com/download/item/7692/OpenSSL_1_0_1l-install-msvc1600-32.tar.gz)
      set(OpenSSL_1.0.1l_1600_MD5 3f756fc731fb95f919ea24f2ec979732)

    #--------------------
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit

      # OpenSSL 1.0.1h
      # VS2008
      set(OpenSSL_1.0.1h_1500_URL http://packages.kitware.com/download/item/6090/OpenSSL_1_0_1h-install-msvc1500-64.tar.gz)
      set(OpenSSL_1.0.1h_1500_MD5 dab0c026ab56fd0fbfe2843d14218fad)
      # VS2010
      set(OpenSSL_1.0.1h_1600_URL http://packages.kitware.com/download/item/6099/OpenSSL_1_0_1h-install-msvc1600-64.tar.gz)
      set(OpenSSL_1.0.1h_1600_MD5 b54a0a4b396397fdf96e55f0f7345dd1)
      # VS2013
      set(OpenSSL_1.0.1h_1800_URL http://packages.kitware.com/download/bitstream/8915/OpenSSL_1_0_1h-install-msvc1800-64.tar.gz)
      set(OpenSSL_1.0.1h_1800_MD5 7aefdd94babefbe603cca48ff86da768)
      # VS2015
      set(OpenSSL_1.0.1h_1900_URL https://packages.kitware.com/download/item/10331/OpenSSL_1_0_1h-install-msvc1900-64.tar.gz)
      set(OpenSSL_1.0.1h_1900_MD5 f93d266def384926015550452573e824)

      # OpenSSL 1.0.1l
      # VS2008
      set(OpenSSL_1.0.1l_1500_URL http://packages.kitware.com/download/item/7686/OpenSSL_1_0_1l-install-msvc1500-64.tar.gz)
      set(OpenSSL_1.0.1l_1500_MD5 ce819931504f8e7c5d9024215b2cc9cf)
      # VS2010
      set(OpenSSL_1.0.1l_1600_URL http://packages.kitware.com/download/item/7700/OpenSSL_1_0_1l-install-msvc1600-64.tar.gz)
      set(OpenSSL_1.0.1l_1600_MD5 07d259e1a467d9c7a7d9cddfd77e8915)

    endif()

    if(NOT DEFINED OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_${MSVC_VERSION}_URL)
      message(FATAL_ERROR ${_error_msg})
    endif()

    #------------------------------------------------------------------------------
    set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

    ExternalProject_Add(${proj}
      ${${proj}_EP_ARGS}
      URL ${OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_${MSVC_VERSION}_URL}
      URL_MD5 ${OpenSSL_${OPENSSL_DOWNLOAD_VERSION}_${MSVC_VERSION}_MD5}
      DOWNLOAD_DIR ${CMAKE_BINARY_DIR}
      SOURCE_DIR ${EP_SOURCE_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      DEPENDS
        ${${proj}_DEPENDENCIES}
      )

    set(OpenSSL_DIR ${EP_SOURCE_DIR})
    set(_openssl_base_dir ${OpenSSL_DIR})
    if(DEFINED CMAKE_CONFIGURATION_TYPES)
      set(OpenSSL_DIR ${OpenSSL_DIR}/${CMAKE_CFG_INTDIR})
      set(_copy_release_directory 1)
    else()
      set(OpenSSL_DIR ${OpenSSL_DIR}/${CMAKE_BUILD_TYPE})
      if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(_copy_release_directory 1)
      endif()
    endif()

    # Support building in RelWithDebInfo configuration
    if(_copy_release_directory)
      ExternalProject_Add_Step(${proj} copy_release_directory
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_openssl_base_dir}/Release" "${_openssl_base_dir}/RelWithDebInfo"
        COMMENT "Copying '${_openssl_base_dir}/Release' to '${_openssl_base_dir}/RelWithDebInfo'"
        DEPENDEES install
        )
    endif()

    set(OPENSSL_INCLUDE_DIR "${OpenSSL_DIR}/include")
    set(OPENSSL_LIBRARY_DIR "${OpenSSL_DIR}/lib")
    set(OPENSSL_EXPORT_LIBRARY_DIR "${OpenSSL_DIR}/bin")

    set(LIB_EAY_DEBUG "${EP_SOURCE_DIR}/Debug/lib/libeay32.lib")
    set(LIB_EAY_RELEASE "${EP_SOURCE_DIR}/Release/lib/libeay32.lib")
    set(SSL_EAY_DEBUG "${EP_SOURCE_DIR}/Debug/lib/ssleay32.lib")
    set(SSL_EAY_RELEASE "${EP_SOURCE_DIR}/Release/lib/ssleay32.lib")

    ExternalProject_Message(${proj} "LIB_EAY_DEBUG:${LIB_EAY_DEBUG}")
    ExternalProject_Message(${proj} "LIB_EAY_RELEASE:${LIB_EAY_RELEASE}")
    ExternalProject_Message(${proj} "SSL_EAY_DEBUG:${SSL_EAY_DEBUG}")
    ExternalProject_Message(${proj} "SSL_EAY_RELEASE:${SSL_EAY_RELEASE}")
  endif()

  ExternalProject_GenerateProjectDescription_Step(${proj}
    VERSION ${OPENSSL_DOWNLOAD_VERSION}
    )

  #-----------------------------------------------------------------------------
  # Launcher setting specific to build tree

  set(${proj}_LIBRARY_PATHS_LAUNCHER_BUILD ${OPENSSL_EXPORT_LIBRARY_DIR})
  mark_as_superbuild(
    VARS ${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    LABELS "LIBRARY_PATHS_LAUNCHER_BUILD"
    )

  ExternalProject_Message(${proj} "OpenSSL ${OPENSSL_DOWNLOAD_VERSION}")
  ExternalProject_Message(${proj} "OPENSSL_LIBRARY_DIR:${OPENSSL_LIBRARY_DIR}")
  ExternalProject_Message(${proj} "OPENSSL_EXPORT_LIBRARY_DIR:${OPENSSL_EXPORT_LIBRARY_DIR}")
else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(
  VARS OPENSSL_INCLUDE_DIR:PATH
  LABELS "FIND_PACKAGE"
  )

# XXX What should we do when OpenSSL is on the system or in a custom location ?
mark_as_superbuild(
  VARS
    OPENSSL_EXPORT_LIBRARY_DIR:PATH
  )

if(UNIX)
  mark_as_superbuild(
    VARS
      OPENSSL_SSL_LIBRARY:FILEPATH
      OPENSSL_CRYPTO_LIBRARY:FILEPATH
    LABELS "FIND_PACKAGE"
    )
  # OPENSSL_LIBRARIES
  set(OPENSSL_LIBRARIES ${OPENSSL_CRYPTO_LIBRARY} ${OPENSSL_SSL_LIBRARY})
  mark_as_superbuild(
    VARS
      OPENSSL_LIBRARIES:STRING
    )
elseif(WIN32)
  mark_as_superbuild(
    VARS
      LIB_EAY_DEBUG:FILEPATH
      LIB_EAY_RELEASE:FILEPATH
      SSL_EAY_DEBUG:FILEPATH
      SSL_EAY_RELEASE:FILEPATH
    LABELS "FIND_PACKAGE"
    )
  # OPENSSL_LIBRARIES
  include(SelectLibraryConfigurations)
  set(LIB_EAY_LIBRARY_DEBUG "${LIB_EAY_DEBUG}")
  set(LIB_EAY_LIBRARY_RELEASE "${LIB_EAY_RELEASE}")
  set(SSL_EAY_LIBRARY_DEBUG "${SSL_EAY_DEBUG}")
  set(SSL_EAY_LIBRARY_RELEASE "${SSL_EAY_RELEASE}")
  select_library_configurations(LIB_EAY)
  select_library_configurations(SSL_EAY)
  set(OPENSSL_LIBRARIES ${SSL_EAY_LIBRARY} ${LIB_EAY_LIBRARY})
  mark_as_superbuild(
    VARS
      OPENSSL_LIBRARIES:STRING
    )
endif()

ExternalProject_Message(${proj} "OPENSSL_INCLUDE_DIR:${OPENSSL_INCLUDE_DIR}")
ExternalProject_Message(${proj} "OPENSSL_LIBRARIES:${OPENSSL_LIBRARIES}")
