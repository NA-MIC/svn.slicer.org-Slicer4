#!/usr/bin/env python

#
#  Program: 3D Slicer
#
#  Copyright (c) Kitware Inc.
#
#  See COPYRIGHT.txt
#  or http://www.slicer.org/copyright/copyright.txt for details.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  This file was originally developed by Johan Andruejol, Kitware Inc.
#

from __future__ import print_function
import argparse
import csv
import json
import os
import tempfile
import sys

from SlicerAppTesting import *

"""
Usage:
    CLISerializationTest.py
      /path/to/Slicer /path/to/CLIExecutables /path/to/data_dir /path/to/temp_dir
"""
class CLISerializationTest(object):
  def __init__(self):
    self.SlicerExecutable = None

  def _runCLI(self, cli_name, option, json_file_path, parameters=[]):
    args = ['--launch',
      cli_name,
      option, json_file_path,
      ]
    args.extend(parameters)
    return run(self.SlicerExecutable, args)

  def serializeCLI(self, cli_name, json_file_path, parameters=[]):
    return self._runCLI(cli_name, '--serialize', json_file_path, parameters)

  def deserializeCLI(self, cli_name, json_file_path, parameters=[]):
    return self._runCLI(cli_name, '--deserialize', json_file_path, parameters)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Test command line CLIs serialization/deserialization.')
  # Common options
  parser.add_argument("/path/to/Slicer")
  parser.add_argument("/path/to/data_dir")
  parser.add_argument("/path/to/temp_dir")
  args = parser.parse_args()

  # Get testing parameters
  slicer_executable = os.path.expanduser(getattr(args, "/path/to/Slicer"))
  data_dir = os.path.expanduser(getattr(args, "/path/to/data_dir"))
  temp_dir = os.path.expanduser(getattr(args, "/path/to/temp_dir"))

  # Create input/output
  serializeSeedsOutFile = '%s/%s.acsv' %(temp_dir, 'SeedsSerialized')
  deserializeSeedsOutFile = '%s/%s.acsv' %(temp_dir, 'SeedsDeSerialized')
  json_file = '%s/%s.json' %(temp_dir, 'ExecutionModelTourSerialized')

  # -- Test ExecutionModelTour --
  EMTSerializer = CLISerializationTest()
  EMTSerializer.SlicerExecutable = slicer_executable
  CLIName = 'ExecutionModelTour'
  required_inputs = [
    '--transform1', '%s/ExecutionModelTourTest.mrml#vtkMRMLLinearTransformNode1'%(data_dir),
    '--transform2', '%s/ExecutionModelTourTest.mrml#vtkMRMLLinearTransformNode2'%(data_dir),
    '%s/MRHeadResampled.nhdr'%(data_dir),
    '%s/CTHeadAxial.nhdr'%(data_dir),
    ]
  serialize_options = [
    '--integer', '30',
    '--double', '30',
    '-f', '1.3,2,-14',
    '--files', '1.does,2.not,3.matter',
    '--string_vector', 'foo,bar,foobar',
    '--enumeration', 'Bill',
    '--boolean1',
    '--seed', '1.0,0.0,-1.0',
    '--seedsOutFile', serializeSeedsOutFile
    ]
  parameters = serialize_options
  parameters.extend(required_inputs)

  # Serialize the CLI
  (returncode, serializeErr, serializeOut) = EMTSerializer.serializeCLI(CLIName, json_file, parameters)
  if returncode != EXIT_SUCCESS:
    print("Problem while serializing the CLI: %s" %serializeErr)
    exit(EXIT_FAILURE)

  # Make sure the Json is generated correctly
  expected_json = {
      "Parameters" :
      {
        "Boolean Parameters" :
        {
          "boolean1" : True,
          "boolean2" : False,
          "boolean3" : False
        },
        "Enumeration Parameters" :
        {
          "stringChoice" : "Bill"
        },
        "File, Directory and Image Parameters" :
        {
          "directory1" : "",
          "file1" : "",
          "files" : [ "1.does", "2.not", "3.matter" ],
          "image1" : "",
          "image2" : "",
          "outputFile1" : ""
        },
        "Generic Tables" :
        {
          "inputDT" : "",
          "outputDT" : ""
        },
        "Geometry Parameters" :
        {
          "InputModel" : "",
          "ModelSceneFile" : [],
          "OutputModel" : ""
        },
        "Index Parameters" :
        {
          "arg0" : "%s/MRHeadResampled.nhdr"%(data_dir),
          "arg1" : "%s/CTHeadAxial.nhdr"%(data_dir)
        },
        "Measurements" :
        {
          "inputFA" : "",
          "outputFA" : ""
        },
        "Point Parameters" :
        {
          "seed" : [[1.0,0.0,-1.0]],
          "seedsFile" : "",
          "seedsOutFile" : serializeSeedsOutFile,
        },
        "Regions of interest" :
        {
          "regions" : []
        },
        "Scalar Parameters" :
        {
          "doubleVariable" : 30,
          "integerVariable" : 30
        },
        "Simple return types" :
        {
          "abooleanreturn" : False,
          "adoublereturn" : 14,
          "afloatreturn" : 7,
          "anintegerreturn" : 5,
          "anintegervectorreturn" : [],
          "astringchoicereturn" : "Bill",
          "astringreturn" : "Hello"
        },
        "Transform Parameters" :
        {
          "transform1" : "%s/ExecutionModelTourTest.mrml#vtkMRMLLinearTransformNode1"%(data_dir),
          "transform2" : "%s/ExecutionModelTourTest.mrml#vtkMRMLLinearTransformNode2"%(data_dir),
          "transformInput" : "",
          "transformInputBspline" : "",
          "transformInputNonlinear" : "",
          "transformOutput" : "",
          "transformOutputBspline" : "",
          "transformOutputNonlinear" : ""
        },
        "Vector Parameters" :
        {
          "floatVector" : [ 1.2999999523162842, 2, -14 ],
          "stringVector" : [ "foo", "bar", "foobar" ]
        }
      }
    }

  with open(json_file) as file:
    data = json.load(file)
    if data != expected_json:
      print('Json comparison failed !')
      exit(EXIT_FAILURE)

  # Now try to deserialize the CLI.
  parameters = [
    '--seedsOutFile', deserializeSeedsOutFile
    ]
  (returncode, deserializeErr, deserializeOut) = EMTSerializer.deserializeCLI(CLIName, json_file, parameters)
  if returncode != EXIT_SUCCESS:
    print("Problem while deserializing the CLI: %s" %deserializeErr)
    exit(EXIT_FAILURE)

  # Finally compare seeds file
  with open(serializeSeedsOutFile) as in_file, open(deserializeSeedsOutFile) as out_file:
    in_reader = csv.reader(in_file)
    out_reader = csv.reader(out_file)

    serializedRows = list(in_reader)
    deserializedRows = list(out_reader)
    if len(serializedRows) != len(deserializedRows):
      print('Seeds comparison failed, files have different number of rows !')
      exit(EXIT_FAILURE)

    for i in range(len(serializedRows)):
      if serializedRows[i] != deserializedRows[i]:
        print('Row #%s comparison failed:' %i)
        print('Serialize row: %s' %serializedRows[i])
        print('Deserialize row: %s' %deserializedRows[i])
        exit(EXIT_FAILURE)

  try:
    os.remove(serializeSeedsOutFile)
    os.remove(deserializeSeedsOutFile)
    os.remove(json_file)
  except AttributeError, OSError:
    pass

  print("\n=> ok")
  exit(EXIT_SUCCESS)
