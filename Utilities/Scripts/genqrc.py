#!/usr/bin/env python

import argparse
import os
import sys

#-----------------------------------------------------------------------------
def writeFile(path, content):
  # Test if file already contains desired content
  if os.path.exists(path):
    try:
      with open(path, "rt") as f:
        if f.read() == content:
          return

    except:
      pass

  # Write file
  with open(path, "wt") as f:
    f.write(content)

#-----------------------------------------------------------------------------
def addFile(path):
  name = os.path.basename(path)
  return ["    <file alias=\"%s\">%s</file>" % (name, path)]

#-----------------------------------------------------------------------------
def buildContent(root, path):
  dirs = []
  out = ["  <qresource prefix=\"%s\">" % path]

  for entry in os.listdir(os.path.join(root, path)):
    full_entry = os.path.join(root, path, entry)

    if os.path.isdir(full_entry):
      dirs.append(os.path.join(path, entry))

    else:
      ext = os.path.splitext(entry)[1].lower()

      if ext == ".png" or ext == ".svg":
        out += addFile(full_entry)

  out += ["  </qresource>", ""]

  for d in dirs:
    out += buildContent(root, d)

  return out

#-----------------------------------------------------------------------------
def main(argv):
  parser = argparse.ArgumentParser(description="PythonQt Resource Compiler")

  parser.add_argument("-o", dest="out_path", metavar="PATH", default="-",
                      help="location to which to write the output .qrc file"
                           " (default=stdout)")
  parser.add_argument("resource_directories", nargs="+",
                      help="list of directories containing resource files")

  args = parser.parse_args(argv)

  qrc_content = [
    "<!DOCTYPE RCC>",
    "<!--",
    "  This file was automatically generated by %s %r" %
      (os.path.basename(__file__), args.resource_directories),
    "  See %s/%s for more information." %
      ("http://www.slicer.org/slicerWiki/index.php",
       "Documentation/Nightly/Developers/Build_system/Qt_resource_files"),
    "-->",
    "",
    "<RCC version=\"1.0\">",
    ""]

  for path in args.resource_directories:
    path = os.path.dirname(os.path.join(path, '.')) # remove trailing '/'
    qrc_content += buildContent(os.path.dirname(path), os.path.basename(path))

  qrc_content += ["</RCC>"]

  qrc_content = "\n".join(qrc_content) + "\n"

  if args.out_path == "-":
    sys.stdout.write(qrc_content)

  else:
    writeFile(args.out_path, qrc_content)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
  main(sys.argv[1:])
