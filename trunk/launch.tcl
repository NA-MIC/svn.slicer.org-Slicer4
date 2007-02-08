#!/bin/sh
# the next line restarts using tclsh \
    exec wish "$0" "$@"

#
# launch.tcl
#
# This script is used to set platform-specific variables for the shell
# that will be used to launch slicer.
#
# - what platform you're on so correct libs are loaded
# - what modules are present so they can be loaded
# - set all PATH, LD_LIBRARY_PATH, TCLLIBPATH, and other shell-level
#   variables to reflect that state
#   
# This is a tcl script rather than a .bat or .sh file so it can handle all 
# the details in one place for all platforms.
#
# This script is used for both the slicer build directory and for the slicer
# install directory.
 

# these are the directories needed for the cpacked install directory:
#/tmp/Slicer3-3.0.2007-02-06-Linux/lib
#/tmp/Slicer3-3.0.2007-02-06-Linux/lib/KWWidgets:
#/tmp/Slicer3-3.0.2007-02-06-Linux/lib/InsightToolkit
#/tmp/Slicer3-3.0.2007-02-06-Linux/lib/Slicer3/tcl/lib

proc launch_InitEnvironment {} {

    # if SLICER_HOME not set, find it relative to the location of this script
    # -- when run by the launcher, this variable will be set
    if { ![info exists ::SLICER_HOME] } {
        set wd [pwd]
        cd [file dirname [info script]]
        set ::SLICER_HOME [pwd]
        cd $wd
    }

    #
    # Note: the local vars file overrides the default values in this script
    # - use it to set your local environment and then your change won't 
    #   be overwritten when this file is updated
    #

    # Check for a custom configuration, setting the environment variables 
    # on the system level
    if {![info exists ::env(SLICER_CUSTOM_CONFIG)]} {
        # set it to simplify futher checking
        set ::env(SLICER_CUSTOM_CONFIG) "false"
    }
    if {$::env(SLICER_CUSTOM_CONFIG) != "true"} {
        set bypass_msg "\n\n(Set environment variable SLICER_CUSTOM_CONFIG to true to bypass this message)"
    }

    # The environment variables that we need to have set for slicer 
    # to start up properly
    set envVars {VTK_DIR VTK_SRC_DIR KWWIDGETS_DIR ITK_BINARY_PATH SANDBOX_BIN_DIR TCL_BIN_DIR TCL_LIB_DIR TEEM_BIN_DIR}
    # Make up a list of the environment variables that haven't been set already,
    # that we need to set
    set envVarsToSet {}
    foreach v $envVars {
        if {![info exists ::env($v)] || $::env($v) == "" } {
            lappend envVarsToSet $v
        }
    }

    # if it is an empty string or doesn't exist, set the LD_LIBRARY_PATH 
    if { ![info exists ::env(LD_LIBRARY_PATH)] ||
            $::env(LD_LIBRARY_PATH) == "" } {
        set ::env(LD_LIBRARY_PATH) " "
    }

    # if it is an empty string or doesn't exist, set the TCLLIBPATH 
    if {[catch {
        if {$::env(TCLLIBPATH) == ""} { 
            set ::env(TCLLIBPATH) " " 
        }} ex]} {
        set ::env(TCLLIBPATH) " " 
    }

    # if it is an empty string or doesn't exist, set the DYLD_LIBRARY_PATH 
    if { $::env(BUILD) == $::DARWIN && [catch {
        if {$::env(DYLD_LIBRARY_PATH) == ""} { 
            set ::env(DYLD_LIBRARY_PATH) " " 
        }} ex]} {
        set ::env(DYLD_LIBRARY_PATH) " "
    }
     
    #
    # VTK source and binary dirs and tcl dirs should be in the Lib directory
    #

    # Set up the VTK and TCL environment variables
    foreach v $envVars {
        if {[lsearch $envVarsToSet $v] != -1} {
            # it's not set already, use our values, either from the variables file, 
            # or our defaults set above, ie set env(VTK_DIR) $VTK_DIR
            if { [namespace eval :: set $v] == "" } {
                # need to set empty string to something because setting
                # env var to "" actually unsets it so it can't be referenced later
                set ::env($v) ":"
            } else {
                set ::env($v) [namespace eval :: set $v]
            }
        } else {
            # it's already been set, don't over-ride
            puts stderr "NOT Overriding current $v $::env($v)"
            if {$::env(SLICER_CUSTOM_CONFIG) != "true"} {
                tk_messageBox -type ok -message "NOT Overriding current $v $::env($v) $bypass_msg"
            }
        }
    }
}


proc launch_SetPaths {} {
    #
    # add the necessary library directories, both Base and Modules, to the 
    # LD_LIBRARY_PATH environment variable
    #
    #
    # set the base library paths for this build 
    # 
    if {$::env(BUILD) == $::SOLARIS ||
        $::env(BUILD) == $::LINUX_64 || 
        $::env(BUILD) == $::LINUX} {
        # add vtk, slicer, and tcl bins
            set ::env(LD_LIBRARY_PATH) $::env(VTK_DIR)/bin:$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(KWWIDGETS_DIR)/bin:$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(ITK_BINARY_PATH)/bin:$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(SANDBOX_BIN_DIR):$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(SANDBOX_BIN_DIR)/../Distributions/bin:$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(SLICER_HOME)/Base/builds/$::env(BUILD)/bin:$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(TCL_LIB_DIR):$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(TCL_BIN_DIR):$::env(LD_LIBRARY_PATH)
            set ::env(LD_LIBRARY_PATH) $::env(TEEM_BIN_DIR):$::env(LD_LIBRARY_PATH)
        } elseif {$::env(BUILD) ==  $::DARWIN} { 
            # add vtk, slicer, and tcl bins
            set ::env(DYLD_LIBRARY_PATH) $::env(VTK_DIR)/bin:$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(KWWIDGETS_DIR)/bin:$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(ITK_BINARY_PATH)/bin:$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(SANDBOX_BIN_DIR):$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(SANDBOX_BIN_DIR)/../Distributions/bin:$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(SLICER_HOME)/Base/builds/$::env(BUILD)/bin:$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(TCL_LIB_DIR):$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(TCL_BIN_DIR):$::env(DYLD_LIBRARY_PATH)
            set ::env(DYLD_LIBRARY_PATH) $::env(TEEM_BIN_DIR):$::env(DYLD_LIBRARY_PATH)
        } elseif {$::env(BUILD) == $::WINDOWS} {
            # add vtk, slicer, and tcl bins
            set ::env(Path) $::env(VTK_DIR)/bin/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(KWWIDGETS_DIR)/bin/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(ITK_BINARY_PATH)/bin/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(SANDBOX_BIN_DIR)/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(SANDBOX_BIN_DIR)/../Distributions/bin/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(SLICER_HOME)/Base/builds/$::env(BUILD)/bin/$::env(VTK_BUILD_SUBDIR)\;$::env(Path)
            set ::env(Path) $::env(TCL_BIN_DIR)\;$::env(Path)
            set ::env(Path) $::env(TEEM_BIN_DIR)\;$::env(Path)
        } else {
            puts stderr "Libraries: unknown build $::env(BUILD)"
        }

    # set the base tcl/tk library paths, using the previously defined TCL_LIB_DIR
    set ::env(TCL_LIBRARY) $::env(TCL_LIB_DIR)/tcl8.4
    set ::env(TK_LIBRARY) $::env(TCL_LIB_DIR)/tk8.4

    #
    # add the default search locations for tcl packages
    #  (window has special tcl packages depending on build type)
    #
    if {$::env(BUILD) == $::SOLARIS || 
        $::env(BUILD) == $::LINUX ||
        $::env(BUILD) == $::LINUX_64 ||
        $::env(BUILD) == $::DARWIN} {
            set ::env(TCLLIBPATH) "$::env(VTK_DIR)/Wrapping/Tcl $::env(TCLLIBPATH)"
            set ::env(TCLLIBPATH) "$::env(KWWIDGETS_DIR)/Wrapping/Tcl $::env(TCLLIBPATH)"
    } elseif {$::env(BUILD) == $::WINDOWS} {
        set ::env(TCLLIBPATH) "$::env(VTK_DIR)/Wrapping/Tcl/$::env(VTK_BUILD_SUBDIR) $::env(TCLLIBPATH)"
        set ::env(TCLLIBPATH) "$::env(KWWIDGETS_DIR)/Wrapping/Tcl/$::env(VTK_BUILD_SUBDIR) $::env(TCLLIBPATH)"
    } else {
        puts stderr "TCLLIBPATH: Invalid build $::env(BUILD)"
        exit
    }

    # same for all platforms
    set ::env(TCLLIBPATH) "$::env(SLICER_HOME)/Base/Wrapping/Tcl/vtkSlicerBase $::env(TCLLIBPATH)"
}


#
# optional - show dialog
#
proc launch_LicenseDialog {} {


    set msg "Slicer is an experimental software package.\nAny clinical use requires proper research controls.\nSelecting \"Ok\" below binds you to the license agreement.\nSee www.slicer.org for details.\n"

    set ::argv0 [lindex $::argv 0]
    if { $::argv0 == "-y" || $::argv0 == "--agree_to_license" || $::argv0 == "--batch" } {
        set ::argv [lreplace $::argv 0 0]
    } else {
        if { ![file exists $::env(HOME)/.IAgreeToSlicersLicense] } {

            if { [catch "package require Tk"] } {
                puts "\n$msg"
                puts -nonewline "Agree? \[Hit Enter for \"Ok\"\] "
                flush stdout
                gets stdin line
                if { $line != "" } {
                    set resp "cancel"
                } else {
                    set resp "ok"
                }
            } else {
                wm withdraw .
                set resp [tk_messageBox -message $msg -type okcancel -title "Slicer3"]
            }

              wm withdraw .
            if {$resp == "cancel"} {
                exit
            }
        }
    }

    if { $::argv0 == "--batch" } {
        set ::BATCH "true"
        puts "Note: By specifying --batch on the command line, you have agreed to slicer's license terms."
    } else {
        set ::BATCH "false"
    }
}

#
# helper proc to handle text coming from the child process
#
proc launch_FileEvent {fp} {
    global ::END
    if {[eof $fp]} {
        catch "close $fp"
        set ::END 1
    } else {
        gets $fp line
        catch {puts stderr $line}
    }
}

proc launch_RunProgram {} {

    #
    # change from tcl escape to shell escape for command line arguments 
    # that contain spaces -- note that shell notation only works for a single
    # level of nesting
    # - change backslashes to forward slashes (for windows paths)
    # - escape the spaces in each of the arguments
    # - then remove the curly braces
    #
    regsub -all "\\\\" $::argv "/" ::argv
    set newargv ""
    foreach a $::argv {
        regsub -all " " $a "\\\ " a
        lappend newargv $a
    }
    set ::argv $newargv


    #
    # if in batch mode, exec vtk with slicer args and return the exit code of the child
    # process as the exit code of this script.
    #
    # Otherwise,
    # launch the slicer main script as a sub-process and monitor the output
    # - process is started with the 'open' command through a pipe
    # - the file_event callback is triggered when subprocess has written to stdout
    # - pipe is closed when process exits, causing launcher to exit too.
    #
    # why? well, some messages only go to console stdout and we don't want to lose them.
    # Also, on windows if you block waiting for the process to exit the launcher would
    # be "Not Responding" and things like screen savers and power management might not 
    # work right.
    #

    # 
    # determine the correct arguments and executable for the platform
    #
    if {$::env(BUILD) == $::WINDOWS} {
        set slicer3 $::SLICER_BUILD/bin/$::env(VTK_BUILD_SUBDIR)/Slicer3.exe
    } else {
        # turn curly braces into quotes
        regsub -all "{|}" $::argv "\\\"" ::argv
        set slicer3 $::SLICER_BUILD/bin/Slicer3
    }

    #
    # run in batch mode to capture all output or
    # run in non-batch mode emit output line by line for feedback
    # - note: use batch mode to capture return code from the process
    #
    if { $::BATCH == "true" } {
        if {$::env(BUILD) == $::WINDOWS} {
            set ret [catch "exec \"$slicer3\" $::argv" res]
        } else {
            set ret [catch "exec $slicer3 $::argv |& cat" res]
        }
        
        # get the actual exit code of the child process
        set code [lindex $::errorCode 2]

        # if the errorCode is not an integer value, set it to 1.
        if { ![string is integer -strict $code] } {
            puts "errorCode is: $code"
            set code 1
            puts "Non-integer error code returned, setting to 1"
        }
        
        # print the stdout/stderr of the child
        puts stdout $res
        # exit with the status
        if { $ret } {
            exit $code
        } else {
            exit 0
        }
    } else {
        if {$::env(BUILD) == $::WINDOWS} {
            set fp [open "| \"$slicer3\" $::argv" r]
        } else {
            set fp [open "| csh -f -c \"$slicer3 $::argv \" |& cat" r]
        }

        if {[info exists fp]} {
            fileevent $fp readable "launch_FileEvent $fp"
            
            set ::END 0
            while { ![catch "pid $fp"] && ![eof $fp] } {
                vwait ::END
            }
        }
    }
}

#######  The actual steps for launching:

launch_InitEnvironment 

launch_SetPaths 

launch_SetupModules 

#launch_LicenseDialog

#
# run program until it exits and then exit this script with the ouput
#
launch_RunProgram



