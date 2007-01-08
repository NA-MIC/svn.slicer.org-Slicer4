// This file was automatically generated by:
//  d:\lorensen\Projects\Slicer3-net\bin\RelWithDebInfo\GenerateCLP.exe GenerateCLP.xml GenerateCLP.h
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>
#include "tclap/CmdLine.h"
#include "ModuleProcessInformation.h"
#include <itksys/ios/sstream>

void
splitString (std::string &text,
             std::string &separators,
             std::vector<std::string> &words)
{
  int n = text.length();
  int start, stop;
  start = text.find_first_not_of(separators);
  while ((start >= 0) && (start < n))
    {
    stop = text.find_first_of(separators, start);
    if ((stop < 0) || (stop > n)) stop = n;
    words.push_back(text.substr(start, stop - start));
    start = text.find_first_not_of(separators, stop+1);
    }
}

#ifdef main
#ifdef WIN32
#define Slicer_EXPORT __declspec(dllexport)
#else
#define Slicer_EXPORT 
#endif

extern "C" {
  Slicer_EXPORT char *GetXMLModuleDescription();
  Slicer_EXPORT int SlicerModuleEntryPoint(int, char*[]);
}
#endif

char *GetXMLModuleDescription()
  {
  std::string xml;
  xml += "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
  xml += "<executable>\n";
  xml += "  <category>utility</category>\n";
  xml += "  <title>NAMIC Command Line Processing Code Generator</title>\n";
  xml += "  <description>Generates C++ code that will parse command lines</description>\n";
  xml += "  <version>1.0</version>\n";
  xml += "  <documentationurl></documentationurl>\n";
  xml += "  <license></license>\n";
  xml += "  <contributor>Bill Lorensen</contributor>\n";
  xml += "\n";
  xml += "  <parameters>\n";
  xml += "    <label>Parameters</label>\n";
  xml += "    <description>Parameters used for command line processing</description>\n";
  xml += "    <boolean>\n";
  xml += "      <name>UseTCLAP</name>\n";
  xml += "      <longflag>--TCLAP</longflag>\n";
  xml += "      <description>Generate TCLAP Code</description>\n";
  xml += "      <label>Generate TCLAP Code</label>\n";
  xml += "      <default>true</default>\n";
  xml += "    </boolean>\n";
  xml += "  </parameters>\n";
  xml += "  <parameters>\n";
  xml += "    <label>IO</label>\n";
  xml += "    <description>Input/Output parameters</description>\n";
  xml += "    <file multiple=\"true\">\n";
  xml += "      <name>logoFiles</name>\n";
  xml += "      <label>Logo Files XML</label>\n";
  xml += "      <longflag>--logoFiles</longflag>\n";
  xml += "      <description>Logo files</description>\n";
  xml += "    </file>\n";
  xml += "    <file>\n";
  xml += "      <name>InputXML</name>\n";
  xml += "      <label>Input XML</label>\n";
  xml += "      <channel>input</channel>\n";
  xml += "      <index>0</index>\n";
  xml += "      <description>XML description of interface</description>\n";
  xml += "    </file>\n";
  xml += "    <file>\n";
  xml += "      <name>OutputCxx</name>\n";
  xml += "      <label>Output C++</label>\n";
  xml += "      <channel>output</channel>\n";
  xml += "      <index>1</index>\n";
  xml += "      <description>C++ Code to process command line arguments</description>\n";
  xml += "    </file>\n";
  xml += "  </parameters>\n";
  xml += "</executable>\n";
  xml += "\n";
  xml += "\n";
  char *xmlChar = new char[xml.size()+1];
  memcpy (xmlChar, xml.c_str(), xml.size());
  xmlChar[xml.size()] = '\0';
  return xmlChar;
  }
#define GENERATE_XML \
  if (argc >= 2 && (strcmp(argv[1],"--xml") == 0)) \
    { \
    std::cout << GetXMLModuleDescription(); \
    return EXIT_SUCCESS; \
    }
#define GENERATE_TCLAP \
    bool UseTCLAP = false; \
    std::vector<std::string> logoFilesTemp; \
    std::vector<std::string> logoFiles; \
    std::string InputXML; \
    std::string OutputCxx; \
    bool echoSwitch = false; \
    bool xmlSwitch = false; \
    std::string processInformationAddressString = "0"; \
try \
  { \
    TCLAP::CmdLine commandLine ( \
      "Generates C++ code that will parse command lines", \
       ' ', \
      "1.0" ); \
 \
      itksys_ios::ostringstream msg; \
    msg.str("");msg << "Generate TCLAP Code (default: " << UseTCLAP << ")"; \
    TCLAP::SwitchArg UseTCLAPArg("", "TCLAP", msg.str(), commandLine, UseTCLAP); \
 \
    msg.str("");msg << "Logo files";    TCLAP::MultiArg<std::string > logoFilesArg("", "logoFiles", msg.str(), 0, "std::vector<std::string>", commandLine); \
 \
    msg.str("");msg << "XML description of interface";    TCLAP::UnlabeledValueArg<std::string> InputXMLArg("InputXML", msg.str(), 1, InputXML, "std::string", commandLine); \
 \
    msg.str("");msg << "C++ Code to process command line arguments";    TCLAP::UnlabeledValueArg<std::string> OutputCxxArg("OutputCxx", msg.str(), 1, OutputCxx, "std::string", commandLine); \
 \
    msg.str("");msg << "Echo the command line arguments (default: " << echoSwitch << ")"; \
    TCLAP::SwitchArg echoSwitchArg("", "echo", msg.str(), commandLine, echoSwitch); \
 \
    msg.str("");msg << "Produce xml description of command line arguments (default: " << xmlSwitch << ")"; \
    TCLAP::SwitchArg xmlSwitchArg("", "xml", msg.str(), commandLine, xmlSwitch); \
 \
    msg.str("");msg << "Address of a structure to store process information (progress, abort, etc.). (default: " << processInformationAddressString << ")"; \
    TCLAP::ValueArg<std::string > processInformationAddressStringArg("", "processinformationaddress", msg.str(), 0, processInformationAddressString, "std::string", commandLine); \
 \
    commandLine.parse ( argc, (char**) argv ); \
    UseTCLAP = UseTCLAPArg.getValue(); \
    logoFilesTemp = logoFilesArg.getValue(); \
    InputXML = InputXMLArg.getValue(); \
    OutputCxx = OutputCxxArg.getValue(); \
    echoSwitch = echoSwitchArg.getValue(); \
    xmlSwitch = xmlSwitchArg.getValue(); \
    processInformationAddressString = processInformationAddressStringArg.getValue(); \
      { /* Assignment for logoFiles */ \
      for (unsigned int _i = 0; _i < logoFilesTemp.size(); _i++) \
        { \
        std::vector<std::string> words; \
        std::vector<std::string> elements; \
        words.clear(); \
      std::string sep(","); \
        splitString(logoFilesTemp[_i], sep, words); \
        for (unsigned int _j= 0; _j < words.size(); _j++) \
          { \
            logoFiles.push_back((words[_j].c_str())); \
          } \
        } \
      } \
      } \
catch ( TCLAP::ArgException e ) \
  { \
  std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; \
  return ( EXIT_FAILURE ); \
    }
#define GENERATE_ECHOARGS \
if (echoSwitch) \
{ \
std::cout << "Command Line Arguments" << std::endl; \
std::cout << "    UseTCLAP: " << UseTCLAP << std::endl; \
for (unsigned int _i= 0; _i < logoFilesTemp.size(); _i++) \
{ \
std::cout << "logoFiles[" << _i << "]: "; \
std::vector<std::string> words; \
words.clear(); \
      std::string sep(","); \
splitString(logoFilesTemp[_i], sep, words); \
for (unsigned int _j= 0; _j < words.size(); _j++) \
{ \
std::cout <<  words[_j] << " "; \
} \
std::cout << std::endl; \
} \
std::cout << "    InputXML: " << InputXML << std::endl; \
std::cout << "    OutputCxx: " << OutputCxx << std::endl; \
std::cout << "    echoSwitch: " << echoSwitch << std::endl; \
std::cout << "    xmlSwitch: " << xmlSwitch << std::endl; \
std::cout << "    processInformationAddressString: " << processInformationAddressString << std::endl; \
}
#define GENERATE_ProcessInformationAddressDecoding \
ModuleProcessInformation *CLPProcessInformation = 0; \
if (processInformationAddressString != "") \
{ \
sscanf(processInformationAddressString.c_str(), "%p", &CLPProcessInformation); \
}
#define PARSE_ARGS GENERATE_XML;GENERATE_TCLAP;GENERATE_ECHOARGS;GENERATE_ProcessInformationAddressDecoding;
