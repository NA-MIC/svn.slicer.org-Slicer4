#ifndef VTKCUDASUPPORT_H_
#define VTKCUDASUPPORT_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaSupport : public vtkObject
{
        vtkTypeRevisionMacro(vtkCudaSupport, vtkObject);

        static vtkCudaSupport *New();

        void PrintSelf(ostream& os, vtkIndent indent);

protected:
        vtkCudaSupport();
        virtual ~vtkCudaSupport();

        int CheckSupportedCudaVersion(int cudaSupport);
};

#endif /*VTKCUDASUPPORT_H_*/
