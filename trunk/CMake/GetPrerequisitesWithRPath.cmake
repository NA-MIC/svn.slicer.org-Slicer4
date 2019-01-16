# - Functions to analyze and list executable file prerequisites.
# This module provides functions to list the .dll, .dylib or .so
# files that an executable or shared library file depends on. (Its
# prerequisites.)
#
# It uses various tools to obtain the list of required shared library files:
#   dumpbin (Windows)
#   objdump (MinGW on Windows)
#   ldd (Linux/Unix)
#   otool (Mac OSX)
# The following functions are provided by this module:
#   get_prerequisites
#   list_prerequisites
#   list_prerequisites_by_glob
#   gp_append_unique
#   is_file_executable
#   gp_item_default_embedded_path
#     (projects can override with gp_item_default_embedded_path_override)
#   gp_resolve_item
#     (projects can override with gp_resolve_item_override)
#   gp_resolve_embedded_item
#     (projects can override with gp_resolve_embedded_item_override)
#   gp_resolved_file_type
#     (projects can override with gp_resolved_file_type_override)
#   gp_file_type
# Requires CMake 2.6 or greater because it uses function, break, return and
# PARENT_SCOPE.
#
#  GET_PREREQUISITES(<target> <prerequisites_var> <exclude_system> <recurse>
#                    <exepath> <dirs>)
# Get the list of shared library files required by <target>. The list in
# the variable named <prerequisites_var> should be empty on first entry to
# this function. On exit, <prerequisites_var> will contain the list of
# required shared library files.
#
# <target> is the full path to an executable file. <prerequisites_var> is the
# name of a CMake variable to contain the results. <exclude_system> must be 0
# or 1 indicating whether to include or exclude "system" prerequisites. If
# <recurse> is set to 1 all prerequisites will be found recursively, if set to
# 0 only direct prerequisites are listed. <exepath> is the path to the top
# level executable used for @executable_path replacment on the Mac. <dirs> is
# a list of paths where libraries might be found: these paths are searched
# first when a target without any path info is given. Then standard system
# locations are also searched: PATH, Framework locations, /usr/lib...
#
#  LIST_PREREQUISITES(<target> [<recurse> [<exclude_system> [<verbose>]]])
# Print a message listing the prerequisites of <target>.
#
# <target> is the name of a shared library or executable target or the full
# path to a shared library or executable file. If <recurse> is set to 1 all
# prerequisites will be found recursively, if set to 0 only direct
# prerequisites are listed. <exclude_system> must be 0 or 1 indicating whether
# to include or exclude "system" prerequisites. With <verbose> set to 0 only
# the full path names of the prerequisites are printed, set to 1 extra
# informatin will be displayed.
#
#  LIST_PREREQUISITES_BY_GLOB(<glob_arg> <glob_exp>)
# Print the prerequisites of shared library and executable files matching a
# globbing pattern. <glob_arg> is GLOB or GLOB_RECURSE and <glob_exp> is a
# globbing expression used with "file(GLOB" or "file(GLOB_RECURSE" to retrieve
# a list of matching files. If a matching file is executable, its prerequisites
# are listed.
#
# Any additional (optional) arguments provided are passed along as the
# optional arguments to the list_prerequisites calls.
#
#  GP_APPEND_UNIQUE(<list_var> <value>)
# Append <value> to the list variable <list_var> only if the value is not
# already in the list.
#
#  IS_FILE_EXECUTABLE(<file> <result_var>)
# Return 1 in <result_var> if <file> is a binary executable, 0 otherwise.
#
# GP_IS_FILE_EXECUTABLE_EXCLUDE_REGEX can be set to a regular expression used
# to give a hint to identify more quickly if a given file is an executable or not.
# This is particularly useful on unix platform where it can avoid a lot of
# time-consuming call to "file" external process. For packages bundling hundreds
# of libraries, executables, resources and data, it largely speeds up the function
# "get_bundle_all_executables".
# On unix, a convenient command line allowing to collect recursively all file extensions
# useful to generate a regular expression like "\\.(dylib|py|pyc|so)$" is:
#   find . -type f -name '*.*' | sed 's@.*/.*\.@@' | sort | uniq | tr "\\n" "|"
#
#  GP_ITEM_DEFAULT_EMBEDDED_PATH(<item> <default_embedded_path_var>)
# Return the path that others should refer to the item by when the item
# is embedded inside a bundle.
#
# Override on a per-project basis by providing a project-specific
# gp_item_default_embedded_path_override function.
#
#  GP_RESOLVE_ITEM(<context> <item> <exepath> <dirs> <resolved_item_var>)
# Resolve an item into an existing full path file.
#
# Override on a per-project basis by providing a project-specific
# gp_resolve_item_override function.
#
#  GP_RESOLVE_EMBEDDED_ITEM(<context> <embedded_item> <exepath> <resolved_embedded_item_var>)
# Resolve an embedded item into the full path within the full path. Since the item can be
# copied later, it doesn't have to exist when calling this function.
#
# Override on a per-project basis by providing a project-specific
# gp_resolve_embedded_item_override function.
#
# If GP_RPATH_DIR variable is set then item matching '@rpath' are
# resolved using the provided directory. Currently setting this variable
# has an effect only on MacOSX when fixing up application bundle. The directory
# are also assumed to be located within the application bundle. It is
# usually the directory passed to the 'rpath' linker option.
#
#  GP_RESOLVED_FILE_TYPE(<original_file> <file> <exepath> <dirs> <type_var>)
# Return the type of <file> with respect to <original_file>. String
# describing type of prerequisite is returned in variable named <type_var>.
#
# Use <exepath> and <dirs> if necessary to resolve non-absolute <file>
# values -- but only for non-embedded items.
#
# Possible types are:
#   system
#   local
#   embedded
#   other
# Override on a per-project basis by providing a project-specific
# gp_resolved_file_type_override function.
#
#  GP_FILE_TYPE(<original_file> <file> <type_var>)
# Return the type of <file> with respect to <original_file>. String
# describing type of prerequisite is returned in variable named <type_var>.
#
# Possible types are:
#   system
#   local
#   embedded
#   other

#=============================================================================
# Copyright 2008-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

function(gp_append_unique list_var value)
  set(contains 0)

  foreach(item ${${list_var}})
    if("${item}" STREQUAL "${value}")
      set(contains 1)
      break()
    endif()
  endforeach()

  if(NOT contains)
    set(${list_var} ${${list_var}} "${value}" PARENT_SCOPE)
  endif()
endfunction()


function(is_file_executable file result_var)
  #
  # A file is not executable until proven otherwise:
  #
  set(${result_var} 0 PARENT_SCOPE)

  get_filename_component(file_full "${file}" ABSOLUTE)
  string(TOLOWER "${file_full}" file_full_lower)

  # If file name ends in .exe on Windows, *assume* executable:
  #
  if(WIN32 AND NOT UNIX)
    if("${file_full_lower}" MATCHES "\\.exe$")
      set(${result_var} 1 PARENT_SCOPE)
      return()
    endif()

    # A clause could be added here that uses output or return value of dumpbin
    # to determine ${result_var}. In 99%+? practical cases, the exe name
    # match will be sufficient...
    #
  endif()

  # Use the information returned from the Unix shell command "file" to
  # determine if ${file_full} should be considered an executable file...
  #
  # If the file command's output contains "executable" and does *not* contain
  # "text" then it is likely an executable suitable for prerequisite analysis
  # via the get_prerequisites macro.
  #
  if(UNIX)

    if(NOT "${GP_IS_FILE_EXECUTABLE_EXCLUDE_REGEX}" STREQUAL "")
      if(${file_full} MATCHES "${GP_IS_FILE_EXECUTABLE_EXCLUDE_REGEX}")
        set(${result_var} 0 PARENT_SCOPE)
        return()
      endif()
    endif()

    if(NOT file_cmd)
      find_program(file_cmd "file")
      mark_as_advanced(file_cmd)
    endif()

    if(file_cmd)
      execute_process(COMMAND "${file_cmd}" "${file_full}"
        OUTPUT_VARIABLE file_ov
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

      # Replace the name of the file in the output with a placeholder token
      # (the string " _file_full_ ") so that just in case the path name of
      # the file contains the word "text" or "executable" we are not fooled
      # into thinking "the wrong thing" because the file name matches the
      # other 'file' command output we are looking for...
      #
      string(REPLACE "${file_full}" " _file_full_ " file_ov "${file_ov}")
      string(TOLOWER "${file_ov}" file_ov)

      #message(STATUS "file_ov='${file_ov}'")
      if("${file_ov}" MATCHES "executable")
        #message(STATUS "executable!")
        if("${file_ov}" MATCHES "text")
          #message(STATUS "but text, so *not* a binary executable!")
        else()
          set(${result_var} 1 PARENT_SCOPE)
          return()
        endif()
      endif()

      # Also detect position independent executables on Linux,
      # where "file" gives "shared object ... (uses shared libraries)"
      if("${file_ov}" MATCHES "shared object.*\(uses shared libs\)")
        set(${result_var} 1 PARENT_SCOPE)
        return()
      endif()

    else()
      message(STATUS "warning: No 'file' command, skipping execute_process...")
    endif()
  endif()
endfunction()


function(gp_item_default_embedded_path item default_embedded_path_var)

  # On Windows and Linux, "embed" prerequisites in the same directory
  # as the executable by default:
  #
  set(path "@executable_path")
  set(overridden 0)

  # On the Mac, relative to the executable depending on the type
  # of the thing we are embedding:
  #
  if(APPLE)
    #
    # The assumption here is that all executables in the bundle will be
    # in same-level-directories inside the bundle. The parent directory
    # of an executable inside the bundle should be MacOS or a sibling of
    # MacOS and all embedded paths returned from here will begin with
    # "@executable_path/../" and will work from all executables in all
    # such same-level-directories inside the bundle.
    #

    # By default, embed things right next to the main bundle executable:
    #
    set(path "@executable_path/../../Contents/MacOS")

    # Embed .dylibs right next to the main bundle executable:
    #
    if(item MATCHES "\\.dylib$")
      set(path "@executable_path/../MacOS")
      set(overridden 1)
    endif()

    # Embed frameworks in the embedded "Frameworks" directory (sibling of MacOS):
    #
    if(NOT overridden)
      if(item MATCHES "[^/]+\\.framework/")
        set(path "@executable_path/../Frameworks")
        set(overridden 1)
      endif()
    endif()
  endif()

  # Provide a hook so that projects can override the default embedded location
  # of any given library by whatever logic they choose:
  #
  if(COMMAND gp_item_default_embedded_path_override)
    gp_item_default_embedded_path_override("${item}" path)
  endif()

  set(${default_embedded_path_var} "${path}" PARENT_SCOPE)
endfunction()


function(gp_resolve_item context item exepath dirs resolved_item_var)
  set(resolved 0)
  set(resolved_item "${item}")

  # Is it already resolved?
  #
  if(IS_ABSOLUTE "${resolved_item}" AND EXISTS "${resolved_item}")
    set(resolved 1)
  endif()

  if(NOT resolved)
    if(item MATCHES "@executable_path")
      #
      # @executable_path references are assumed relative to exepath
      #
      string(REPLACE "@executable_path" "${exepath}" ri "${item}")
      get_filename_component(ri "${ri}" ABSOLUTE)

      if(EXISTS "${ri}")
        #message(STATUS "info: embedded item exists (${ri})")
        set(resolved 1)
        set(resolved_item "${ri}")
      else()
        message(STATUS "warning: embedded item does not exist '${ri}'")
      endif()
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "@loader_path")
      #
      # @loader_path references are assumed relative to the
      # PATH of the given "context" (presumably another library)
      #
      get_filename_component(contextpath "${context}" PATH)
      string(REPLACE "@loader_path" "${contextpath}" ri "${item}")
      get_filename_component(ri "${ri}" ABSOLUTE)

      if(EXISTS "${ri}")
        #message(STATUS "info: embedded item exists (${ri})")
        set(resolved 1)
        set(resolved_item "${ri}")
      else()
        message(STATUS "warning: embedded item does not exist '${ri}'")
      endif()
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "@rpath")
      #
      # @rpath references are relative to the paths built into the binaries with -rpath
      # We handle this case like we do for other Unixes.
      #
      # Two cases of item resolution are considered:
      #
      #  (1) item has been copied into the bundle
      #
      #  (2) item has NOT been copied into the bundle: Since the item can exist in a build or
      #      install tree outside of the bundle, the item is resolved using its name and the
      #      passed list of directories.
      #
      string(REPLACE "@rpath/" "" norpath_item "${item}")

      set(ri "ri-NOTFOUND")
      if(EXISTS ${GP_RPATH_DIR}/${norpath_item})
        set(ri ${GP_RPATH_DIR}/${norpath_item})
        set(_msg "'find_file' in GP_RPATH_DIR (${ri})")
      else()
        get_filename_component(norpath_item_name ${norpath_item} NAME)
        find_file(ri "${norpath_item_name}" ${exepath} ${dirs} NO_DEFAULT_PATH)
        set(_msg "'find_file' in exepath/dirs (${ri})")
      endif()
      if(ri)
        #message(STATUS "info: ${_msg}")
        set(resolved 1)
        set(resolved_item "${ri}")
        set(ri "ri-NOTFOUND")
      endif()

    endif()
  endif()

  if(NOT resolved)
    set(ri "ri-NOTFOUND")
    find_file(ri "${item}" ${exepath} ${dirs} NO_DEFAULT_PATH)
    find_file(ri "${item}" ${exepath} ${dirs} /usr/lib)
    if(ri)
      #message(STATUS "info: 'find_file' in exepath/dirs (${ri})")
      set(resolved 1)
      set(resolved_item "${ri}")
      set(ri "ri-NOTFOUND")
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "[^/]+\\.framework/")
      set(fw "fw-NOTFOUND")
      find_file(fw "${item}"
        "~/Library/Frameworks"
        "/Library/Frameworks"
        "/System/Library/Frameworks"
      )
      if(fw)
        #message(STATUS "info: 'find_file' found framework (${fw})")
        set(resolved 1)
        set(resolved_item "${fw}")
        set(fw "fw-NOTFOUND")
      endif()
    endif()
  endif()

  # Using find_program on Windows will find dll files that are in the PATH.
  # (Converting simple file names into full path names if found.)
  #
  if(WIN32 AND NOT UNIX)
  if(NOT resolved)
    set(ri "ri-NOTFOUND")
    find_program(ri "${item}" PATHS "${exepath};${dirs}" NO_DEFAULT_PATH)
    find_program(ri "${item}" PATHS "${exepath};${dirs}")
    if(ri)
      #message(STATUS "info: 'find_program' in exepath/dirs (${ri})")
      set(resolved 1)
      set(resolved_item "${ri}")
      set(ri "ri-NOTFOUND")
    endif()
  endif()
  endif()

  # Provide a hook so that projects can override item resolution
  # by whatever logic they choose:
  #
  if(COMMAND gp_resolve_item_override)
    gp_resolve_item_override("${context}" "${item}" "${exepath}" "${dirs}" resolved_item resolved)
  endif()

  if(NOT resolved)
    message(STATUS "
warning: cannot resolve item '${item}'

  possible problems:
    need more directories?
    need to use InstallRequiredSystemLibraries?
    run in install tree instead of build tree?
")
#    message(STATUS "
#******************************************************************************
#warning: cannot resolve item '${item}'
#
#  possible problems:
#    need more directories?
#    need to use InstallRequiredSystemLibraries?
#    run in install tree instead of build tree?
#
#    context='${context}'
#    item='${item}'
#    exepath='${exepath}'
#    dirs='${dirs}'
#    resolved_item_var='${resolved_item_var}'
#******************************************************************************
#")
  endif()

  set(${resolved_item_var} "${resolved_item}" PARENT_SCOPE)
endfunction()

function(gp_resolve_embedded_item context embedded_item exepath resolved_embedded_item_var)
  #message(STATUS "**")
  set(resolved 0)
  set(resolved_embedded_item "${embedded_item}")

  if(embedded_item MATCHES "@executable_path")
    string(REPLACE "@executable_path" "${exepath}" resolved_embedded_item "${embedded_item}")
    set(resolved 1)
  endif()
  if(EXISTS "${GP_RPATH_DIR}" AND embedded_item MATCHES "@rpath")
    string(REPLACE "@rpath" "${GP_RPATH_DIR}" resolved_embedded_item "${embedded_item}")
    set(resolved 1)
  endif()

  # Provide a hook so that projects can override embedded item resolution
  # by whatever logic they choose:
  #
  if(COMMAND gp_resolve_embedded_item_override)
    gp_resolve_embedded_item_override(
      "${context}" "${embedded_item}" "${exepath}" resolved_embedded_item resolved)
  endif()

  if(NOT resolved)
    message(STATUS "
warning: cannot resolve embedded item '${embedded_item}'
  possible problems:
    need more directories?
    need to use InstallRequiredSystemLibraries?
    run in install tree instead of build tree?

    context='${context}'
    embedded_item='${embedded_item}'
    GP_RPATH_DIR='${GP_RPATH_DIR}'
    exepath='${exepath}'
    resolved_embedded_item_var='${resolved_embedded_item_var}'
")
  endif()

  set(${resolved_embedded_item_var} "${resolved_embedded_item}" PARENT_SCOPE)
endfunction()

function(gp_resolved_file_type original_file file exepath dirs type_var)
  #message(STATUS "**")

  if(NOT IS_ABSOLUTE "${original_file}")
    message(STATUS "warning: gp_resolved_file_type expects absolute full path for first arg original_file")
  endif()

  set(is_embedded 0)
  set(is_local 0)
  set(is_system 0)

  set(resolved_file "${file}")

  if("${file}" MATCHES "^@(executable_|loader_|r)path")
    set(is_embedded 1)
  endif()

  if(NOT is_embedded)
    if(NOT IS_ABSOLUTE "${file}")
      gp_resolve_item("${original_file}" "${file}" "${exepath}" "${dirs}" resolved_file)
    endif()

    string(TOLOWER "${original_file}" original_lower)
    string(TOLOWER "${resolved_file}" lower)

    if(UNIX)
      if(resolved_file MATCHES "^(/lib/|/lib32/|/lib64/|/usr/lib/|/usr/lib32/|/usr/lib64/|/usr/X11/|/usr/X11R6/|/usr/bin/|/usr/.*/lib/)")
        set(is_system 1)
      endif()
    endif()

    if(APPLE)
      if(resolved_file MATCHES "^.*Qt.*framework")
        #pass: we need to package Qt
        set(is_system 0)
      elseif(resolved_file MATCHES "^(/System/Library/|/usr/lib/|/opt/X11/)")
        set(is_system 1)
      endif()
    endif()

    if(WIN32)
      string(TOLOWER "$ENV{SystemRoot}" sysroot)
      string(REGEX REPLACE "\\\\" "/" sysroot "${sysroot}")

      string(TOLOWER "$ENV{windir}" windir)
      string(REGEX REPLACE "\\\\" "/" windir "${windir}")

      if(lower MATCHES "^(${sysroot}/sys(tem|wow)|${windir}/sys(tem|wow)|(.*/)*msvc[^/]+dll)")
        set(is_system 1)
      endif()

      if(UNIX)
        # if cygwin, we can get the properly formed windows paths from cygpath
        find_program(CYGPATH_EXECUTABLE cygpath)

        if(CYGPATH_EXECUTABLE)
          execute_process(COMMAND ${CYGPATH_EXECUTABLE} -W
                          OUTPUT_VARIABLE env_windir
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
          execute_process(COMMAND ${CYGPATH_EXECUTABLE} -S
                          OUTPUT_VARIABLE env_sysdir
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
          string(TOLOWER "${env_windir}" windir)
          string(TOLOWER "${env_sysdir}" sysroot)

          if(lower MATCHES "^(${sysroot}/sys(tem|wow)|${windir}/sys(tem|wow)|(.*/)*msvc[^/]+dll)")
            set(is_system 1)
          endif()
        endif()
      endif()
    endif()

    if(NOT is_system)
      get_filename_component(original_path "${original_lower}" PATH)
      get_filename_component(path "${lower}" PATH)
      if("${original_path}" STREQUAL "${path}")
        set(is_local 1)
      else()
        string(LENGTH "${original_path}/" original_length)
        string(LENGTH "${lower}" path_length)
        if(${path_length} GREATER ${original_length})
          string(SUBSTRING "${lower}" 0 ${original_length} path)
          if("${original_path}/" STREQUAL "${path}")
            set(is_embedded 1)
          endif()
        endif()
      endif()
    endif()
  endif()

  # Return type string based on computed booleans:
  #
  set(type "other")

  if(is_system)
    set(type "system")
  elseif(is_embedded)
    set(type "embedded")
  elseif(is_local)
    set(type "local")
  endif()

  #message(STATUS "gp_resolved_file_type: '${file}' '${resolved_file}'")
  #message(STATUS "                type: '${type}'")

  if(NOT is_embedded)
    if(NOT IS_ABSOLUTE "${resolved_file}")
      if(lower MATCHES "^msvc[^/]+dll" AND is_system)
        message(STATUS "info: non-absolute msvc file '${file}' returning type '${type}'")
      else()
        message(STATUS "warning: gp_resolved_file_type non-absolute file '${file}' returning type '${type}' -- possibly incorrect")
      endif()
    endif()
  endif()

  # Provide a hook so that projects can override the decision on whether a
  # library belongs to the system or not by whatever logic they choose:
  #
  if(COMMAND gp_resolved_file_type_override)
    gp_resolved_file_type_override("${resolved_file}" type)
  endif()

  set(${type_var} "${type}" PARENT_SCOPE)

  #message(STATUS "**")
endfunction()


function(gp_file_type original_file file type_var)
  if(NOT IS_ABSOLUTE "${original_file}")
    message(STATUS "warning: gp_file_type expects absolute full path for first arg original_file")
  endif()

  get_filename_component(exepath "${original_file}" PATH)

  set(type "")
  gp_resolved_file_type("${original_file}" "${file}" "${exepath}" "" type)

  set(${type_var} "${type}" PARENT_SCOPE)
endfunction()


function(get_prerequisites target prerequisites_var exclude_system recurse exepath dirs)
  set(verbose 0)
  set(eol_char "E")

  if(NOT IS_ABSOLUTE "${target}")
    message("warning: target '${target}' is not absolute...")
  endif()

  if(NOT EXISTS "${target}")
    message("warning: target '${target}' does not exist...")
  endif()

  set(gp_cmd_paths ${gp_cmd_paths}
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin"
    "C:/Program Files/Microsoft Visual Studio 8/VC/BIN"
    "C:/Program Files (x86)/Microsoft Visual Studio 8/VC/BIN"
    "C:/Program Files/Microsoft Visual Studio .NET 2003/VC7/BIN"
    "C:/Program Files (x86)/Microsoft Visual Studio .NET 2003/VC7/BIN"
    "/usr/local/bin"
    "/usr/bin"
    )

  # <setup-gp_tool-vars>
  #
  # Try to choose the right tool by default. Caller can set gp_tool prior to
  # calling this function to force using a different tool.
  #
  if("${gp_tool}" STREQUAL "")
    set(gp_tool "ldd")

    if(APPLE)
      set(gp_tool "otool")
    endif()

    if(WIN32 AND NOT UNIX) # This is how to check for cygwin, har!
      find_program(gp_dumpbin "dumpbin" PATHS ${gp_cmd_paths})
      if(gp_dumpbin)
        set(gp_tool "dumpbin")
      else() # Try harder. Maybe we're on MinGW
        set(gp_tool "objdump")
      endif()
    endif()
  endif()

  find_program(gp_cmd ${gp_tool} PATHS ${gp_cmd_paths})

  if(NOT gp_cmd)
    message(STATUS "warning: could not find '${gp_tool}' - cannot analyze prerequisites...")
    return()
  endif()

  set(gp_tool_known 0)

  if("${gp_tool}" STREQUAL "ldd")
    set(gp_cmd_args "")
    set(gp_regex "^[\t ]*[^\t ]+ => ([^\t\(]+) .*${eol_char}$")
    set(gp_regex_error "not found${eol_char}$")
    set(gp_regex_fallback "^[\t ]*([^\t ]+) => ([^\t ]+).*${eol_char}$")
    set(gp_regex_cmp_count 1)
    set(gp_tool_known 1)
  endif()

  if("${gp_tool}" STREQUAL "otool")
    set(gp_cmd_args "-L")
    set(gp_regex "^\t([^\t]+) \\(compatibility version ([0-9]+.[0-9]+.[0-9]+), current version ([0-9]+.[0-9]+.[0-9]+)\\)${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 3)
    set(gp_tool_known 1)
  endif()

  if("${gp_tool}" STREQUAL "dumpbin")
    set(gp_cmd_args "/dependents")
    set(gp_regex "^    ([^ ].*[Dd][Ll][Ll])${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 1)
    set(gp_tool_known 1)
    set(ENV{VS_UNICODE_OUTPUT} "") # Block extra output from inside VS IDE.
  endif()

  if("${gp_tool}" STREQUAL "objdump")
    set(gp_cmd_args "-p")
    set(gp_regex "^\t*DLL Name: (.*\\.[Dd][Ll][Ll])${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 1)
    set(gp_tool_known 1)
  endif()

  if(NOT gp_tool_known)
    message(STATUS "warning: gp_tool='${gp_tool}' is an unknown tool...")
    message(STATUS "CMake function get_prerequisites needs more code to handle '${gp_tool}'")
    message(STATUS "Valid gp_tool values are dumpbin, ldd, objdump and otool.")
    return()
  endif()


  if("${gp_tool}" STREQUAL "dumpbin")
    # When running dumpbin, it also needs the "Common7/IDE" directory in the
    # PATH. It will already be in the PATH if being run from a Visual Studio
    # command prompt. Add it to the PATH here in case we are running from a
    # different command prompt.
    #
    get_filename_component(gp_cmd_dir "${gp_cmd}" PATH)
    get_filename_component(gp_cmd_dlls_dir "${gp_cmd_dir}/../../Common7/IDE" ABSOLUTE)
    # Use cmake paths as a user may have a PATH element ending with a backslash.
    # This will escape the list delimiter and create havoc!
    if(EXISTS "${gp_cmd_dlls_dir}")
      # only add to the path if it is not already in the path
      set(gp_found_cmd_dlls_dir 0)
      file(TO_CMAKE_PATH "$ENV{PATH}" env_path)
      foreach(gp_env_path_element ${env_path})
        if("${gp_env_path_element}" STREQUAL "${gp_cmd_dlls_dir}")
          set(gp_found_cmd_dlls_dir 1)
        endif()
      endforeach()

      if(NOT gp_found_cmd_dlls_dir)
        file(TO_NATIVE_PATH "${gp_cmd_dlls_dir}" gp_cmd_dlls_dir)
        set(ENV{PATH} "$ENV{PATH};${gp_cmd_dlls_dir}")
      endif()
    endif()
  endif()
  #
  # </setup-gp_tool-vars>

  if("${gp_tool}" STREQUAL "ldd")
    set(old_ld_env "$ENV{LD_LIBRARY_PATH}")
    foreach(dir ${exepath} ${dirs})
      set(ENV{LD_LIBRARY_PATH} "${dir}:$ENV{LD_LIBRARY_PATH}")
    endforeach()
  endif()


  # Track new prerequisites at each new level of recursion. Start with an
  # empty list at each level:
  #
  set(unseen_prereqs)

  # Run gp_cmd on the target:
  #
  execute_process(
    COMMAND ${gp_cmd} ${gp_cmd_args} ${target}
    OUTPUT_VARIABLE gp_cmd_ov
    )

  if("${gp_tool}" STREQUAL "ldd")
    set(ENV{LD_LIBRARY_PATH} "${old_ld_env}")
  endif()

  if(verbose)
    message(STATUS "<RawOutput cmd='${gp_cmd} ${gp_cmd_args} ${target}'>")
    message(STATUS "gp_cmd_ov='${gp_cmd_ov}'")
    message(STATUS "</RawOutput>")
  endif()

  get_filename_component(target_dir "${target}" PATH)

  # Convert to a list of lines:
  #
  string(REGEX REPLACE ";" "\\\\;" candidates "${gp_cmd_ov}")
  string(REGEX REPLACE "\n" "${eol_char};" candidates "${candidates}")

  # check for install id and remove it from list, since otool -L can include a
  # reference to itself
  set(gp_install_id)
  if("${gp_tool}" STREQUAL "otool")
    execute_process(
      COMMAND otool -D ${target}
      OUTPUT_VARIABLE gp_install_id_ov
      )
    # second line is install name
    string(REGEX REPLACE ".*:\n" "" gp_install_id "${gp_install_id_ov}")
    if(gp_install_id)
      # trim
      string(REGEX MATCH "[^\n ].*[^\n ]" gp_install_id "${gp_install_id}")
      #message("INSTALL ID is \"${gp_install_id}\"")
    endif()
  endif()

  # Analyze each line for file names that match the regular expression:
  #
  foreach(candidate ${candidates})
  if("${candidate}" MATCHES "${gp_regex}")

    # Extract information from each candidate:
    if(gp_regex_error AND "${candidate}" MATCHES "${gp_regex_error}")
      string(REGEX REPLACE "${gp_regex_fallback}" "\\1" raw_item "${candidate}")
    else()
      string(REGEX REPLACE "${gp_regex}" "\\1" raw_item "${candidate}")
    endif()

    if(gp_regex_cmp_count GREATER 1)
      string(REGEX REPLACE "${gp_regex}" "\\2" raw_compat_version "${candidate}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\1" compat_major_version "${raw_compat_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\2" compat_minor_version "${raw_compat_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\3" compat_patch_version "${raw_compat_version}")
    endif()

    if(gp_regex_cmp_count GREATER 2)
      string(REGEX REPLACE "${gp_regex}" "\\3" raw_current_version "${candidate}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\1" current_major_version "${raw_current_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\2" current_minor_version "${raw_current_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\3" current_patch_version "${raw_current_version}")
    endif()

    # Use the raw_item as the list entries returned by this function. Use the
    # gp_resolve_item function to resolve it to an actual full path file if
    # necessary.
    #
    set(item "${raw_item}")

    # Add each item unless it is excluded:
    #
    set(add_item 1)

    if("${item}" STREQUAL "${gp_install_id}")
      set(add_item 0)
    endif()

    if(add_item AND ${exclude_system})
      set(type "")
      gp_resolved_file_type("${target}" "${item}" "${exepath}" "${dirs}" type)

      if("${type}" STREQUAL "system")
        set(add_item 0)
      endif()
    endif()

    if(add_item)
      list(LENGTH ${prerequisites_var} list_length_before_append)
      gp_append_unique(${prerequisites_var} "${item}")
      list(LENGTH ${prerequisites_var} list_length_after_append)

      if(${recurse})
        # If item was really added, this is the first time we have seen it.
        # Add it to unseen_prereqs so that we can recursively add *its*
        # prerequisites...
        #
        # But first: resolve its name to an absolute full path name such
        # that the analysis tools can simply accept it as input.
        #
        if(NOT list_length_before_append EQUAL list_length_after_append)
          gp_resolve_item("${target}" "${item}" "${exepath}" "${dirs}" resolved_item)
          set(unseen_prereqs ${unseen_prereqs} "${resolved_item}")
        endif()
      endif()
    endif()
  else()
    if(verbose)
      message(STATUS "ignoring non-matching line: '${candidate}'")
    endif()
  endif()
  endforeach()

  list(LENGTH ${prerequisites_var} prerequisites_var_length)
  if(prerequisites_var_length GREATER 0)
    list(SORT ${prerequisites_var})
  endif()
  if(${recurse})
    set(more_inputs ${unseen_prereqs})
    foreach(input ${more_inputs})
      get_prerequisites("${input}" ${prerequisites_var} ${exclude_system} ${recurse} "${exepath}" "${dirs}")
    endforeach()
  endif()

  set(${prerequisites_var} ${${prerequisites_var}} PARENT_SCOPE)
endfunction()


function(list_prerequisites target)
  if("${ARGV1}" STREQUAL "")
    set(all 1)
  else()
    set(all "${ARGV1}")
  endif()

  if("${ARGV2}" STREQUAL "")
    set(exclude_system 0)
  else()
    set(exclude_system "${ARGV2}")
  endif()

  if("${ARGV3}" STREQUAL "")
    set(verbose 0)
  else()
    set(verbose "${ARGV3}")
  endif()

  set(count 0)
  set(count_str "")
  set(print_count "${verbose}")
  set(print_prerequisite_type "${verbose}")
  set(print_target "${verbose}")
  set(type_str "")

  get_filename_component(exepath "${target}" PATH)

  set(prereqs "")
  get_prerequisites("${target}" prereqs ${exclude_system} ${all} "${exepath}" "")

  if(print_target)
    message(STATUS "File '${target}' depends on:")
  endif()

  foreach(d ${prereqs})
    math(EXPR count "${count} + 1")

    if(print_count)
      set(count_str "${count}. ")
    endif()

    if(print_prerequisite_type)
      gp_file_type("${target}" "${d}" type)
      set(type_str " (${type})")
    endif()

    message(STATUS "${count_str}${d}${type_str}")
  endforeach()
endfunction()


function(list_prerequisites_by_glob glob_arg glob_exp)
  message(STATUS "=============================================================================")
  message(STATUS "List prerequisites of executables matching ${glob_arg} '${glob_exp}'")
  message(STATUS "")
  file(${glob_arg} file_list ${glob_exp})
  foreach(f ${file_list})
    is_file_executable("${f}" is_f_executable)
    if(is_f_executable)
      message(STATUS "=============================================================================")
      list_prerequisites("${f}" ${ARGN})
      message(STATUS "")
    endif()
  endforeach()
endfunction()
