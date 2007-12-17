#=auto==========================================================================
#   Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.
# 
#   See Doc/copyright/copyright.txt
#   or http://www.slicer.org/copyright/copyright.txt for details.
# 
#   Program:   3D Slicer
#   Module:    $RCSfile: trackerd.tcl,v $
#   Date:      $Date: 2006/01/30 20:47:46 $
#   Version:   $Revision: 1.10 $
# 
#===============================================================================
# FILE:        trackerd.tcl
# PROCEDURES:  
#==========================================================================auto=

#
# experimental tracker daemon - sp 2007-12-14
# - based on slicerd code
#

#
# returns a listening socket on given port or uses default
# - local only for security, use ssh tunnels for remote access
#
proc trackerd_start { {port 8529} } {

    set ::SLICERD(port) $port
    set ret [ catch {set ::SLICERD(serversock) [socket -server trackerd_sock_cb $port]} res]

    if { $ret } {
        puts "Warning: could not start tracker daemon at default port (probably another tracker daemon already running on this machine)."
    }
}

#
# shuts down the socket
# - frees the tcl helper if it exists
#
proc trackerd_stop { } {

    if { ![info exists SLICERD(serversock)] } {
        return
    }
    set _tcl ::tcl_$SLICERD(serversock)
    catch "$_tcl Delete"
    catch "unset ::SLICERD(approved)"

    close $sock
}

#
# accepts new connections
#
proc trackerd_sock_cb { sock addr port } {

    if { ![info exists ::SLICERD(approved)] } {
        set ::SLICERD(approved) [tk_messageBox \
                                     -type yesno \
                                     -title "Slicer Daemon" \
                                     -message "Connection Attemped from $addr.\n\nAllow external connections?"]
    }
    if { $::SLICERD(approved)  == "no" } {
        close $sock
        return
    }

    #
    # create a tcl helper for this socket
    # then set up a callback for when the socket becomes readable
    #
    set _tcl ::tcl_$sock
    catch "$_tcl Delete"
    vtkTclHelper $_tcl
    
    # special trick to let the tcl helper know what interp to use
    set tag [$_tcl AddObserver ModifiedEvent ""]
    $_tcl SetInterpFromCommand $tag

    fileevent $sock readable "trackerd_sock_fileevent $sock"
}

#
# handles input on the connection
#
proc trackerd_sock_fileevent {sock} {
    
    if { [eof $sock] } {
        close $sock
        set _tcl ::tcl_$sock
        catch "$_tcl Delete"
        return
    }

puts "reading..."

    set nodes [$::slicer3::MRMLScene GetNodesByName "tracker"]
    set transformNode [$nodes GetItemAsObject 0]
    if { $transformNode == "" } {
      error "No node named tracker"
    }

    set matrix [$transformNode GetMatrixTransformToParent]
    
    ::tcl_$sock SetMatrix $matrix

    fconfigure $sock -translation binary
    ::tcl_$sock ReceiveMatrix $sock
    fconfigure $sock -translation auto

puts "finished reading!"

}

