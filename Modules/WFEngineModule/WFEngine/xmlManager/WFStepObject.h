#ifndef WFSTEPOBJECT_H_
#define WFSTEPOBJECT_H_

#include <string>
#include <map>
#include <vector>

namespace WFEngine
{
    namespace nmWFStepObject
    {
        class WFStepObject
        {
        public:
            static WFStepObject* New();
            
            struct variablePropertyStruct {
                bool isInputParameter;
                bool isOutputParameter;
                std::string value;
                std::string name;
                std::string type;
                std::string mapsTo;
                std::string decomposesTo;
            };
            
            void SetID(std::string &ID);
            void SetName(std::string &name);
            void AddNextStepID(std::string &nextStepID);
//            void SetVariableMapping(std::string &from, std::string &to);
            void SetDescription(std::string &stepDesc);            
            void AddVariable(std::string &varName, variablePropertyStruct *propStruct);
            
            std::string GetNextStepID();
            std::string GetID();
            std::string GetName();
            std::string GetDescription();
            std::string GetGUIDescription();
            bool ExistsEvent(std::string &eventName);                        
            
        protected:
            WFStepObject();
            virtual ~WFStepObject();
        private:
            std::string m_ID;
            std::string m_name;
            std::vector<std::string> m_nextSteps;
//            std::map<std::string, std::string> m_varMapping;
            std::map<std::string, variablePropertyStruct*> m_varMap;
            std::string m_desc;
            std::string m_guiDesc;
        };
    }
}

#endif /*WFSTEPOBJECT_H_*/
