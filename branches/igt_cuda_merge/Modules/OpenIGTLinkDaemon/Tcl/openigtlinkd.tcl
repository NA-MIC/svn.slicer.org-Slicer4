#=auto==========================================================================
#   Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.
# 
#   See Doc/copyright/copyright.txt
#   or http://www.slicer.org/copyright/copyright.txt for details.
# 
#   Program:   3D Slicer
#   Module:    $RCSfile: openigtlinkd.tcl,v $
#   Date:      $Date: 2006/01/30 20:47:46 $
#   Version:   $Revision: 1.10 $
# 
#===============================================================================
# FILE:        openigtlinkd.tcl
# PROCEDURES:  
#==========================================================================auto=

#
# OpenIGTLink Daemon based on Slicer Daemon
#

#
# returns a listening socket on given port or uses default
# - local only for security, use ssh tunnels for remote access
#
proc openigtlinkd_start { {port 18944} } {

    set ::OPENIGTLINKD(port) $port
    set ret [ catch {set ::OPENIGTLINKD(serversock) [socket -server openigtlinkd_sock_cb $port]} res]

    if { $ret } {
        puts "Warning: could not start OpenIGTLink daemon at default port (probably another slicer daemon already running on this machine)."
    }

}

#
# shuts down the socket
# - frees the tcl helper if it exists
#
proc openigtlinkd_stop { } {

    if { ![info exists OPENIGTLINKD(serversock)] } {
        return
    }
    set _tcl ::tcl_$OPENIGTLINKD(serversock)
    catch "$_tcl Delete"
    catch "unset ::OPENIGTLINKD(approved)"

    close $sock
}

#
# accepts new connections
#
proc openigtlinkd_sock_cb { sock addr port } {

    if { ![info exists ::OPENIGTLINKD(approved)] } {
        set ::OPENIGTLINKD(approved) [tk_messageBox \
                                     -type yesno \
                                     -title "OpenIGTLink Daemon" \
                                     -message "Connection Attemped from $addr.\n\nAllow external connections?"]
    }
    if { $::OPENIGTLINKD(approved)  == "no" } {
        close $sock
        return
    }
    
    #
    # create a tcl helper for this socket
    # then set up a callback for when the socket becomes readable
    #
    set _tcl ::tcl_$sock
    catch "$_tcl Delete"
    vtkOpenIGTLinkTclHelper $_tcl
    
    # special trick to let the tcl helper know what interp to use
    set tag [$_tcl AddObserver ModifiedEvent ""]
    $_tcl SetInterpFromCommand $tag

    fileevent $sock readable "openigtlinkd_sock_fileevent $sock"
}

#
# handles input on the connection
#
proc openigtlinkd_sock_fileevent {sock} {

    if { [eof $sock] } {
        close $sock
        set _tcl ::tcl_$sock
        catch "$_tcl Delete"
        return
    }

    fconfigure $sock -translation binary
    #::tcl_$sock SetScene $::slicer3::MRMLScene 
    #::tcl_$sock SetScene [$::slicer3::ApplicationLogic GetMRMLScene]
    ::tcl_$sock SetAppLogic [$::slicer3::ApplicationGUI GetApplicationLogic]
    ::tcl_$sock OnReceiveOpenIGTLinkMessage $sock
    flush $sock

}
