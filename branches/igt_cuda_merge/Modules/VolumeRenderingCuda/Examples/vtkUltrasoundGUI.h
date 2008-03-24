#ifndef __ULTRASOUND_EXAMPLE_GUI_H_
#define __ULTRASOUND_EXAMPLE_GUI_H_

#include "vtkKWWindow.h"

class vtkUltrasoundGUI : public vtkKWWindow
{
    vtkTypeRevisionMacro(vtkUltrasoundGUI, vtkKWWindow);
    static vtkUltrasoundGUI *New();


protected:
    vtkUltrasoundGUI();
    virtual ~vtkUltrasoundGUI();

    private:
    vtkUltrasoundGUI operator=(const vtkUltrasoundGUI&);
    vtkUltrasoundGUI(const vtkUltrasoundGUI&);

};

#endif /* __ULTRASOUND_EXAMPLE_GUI_H_ */
