#ifndef __vtkQueryAtlasIcons_h
#define __vtkQueryAtlasIcons_h

#include "vtkKWObject.h"
#include "vtkKWResourceUtilities.h"
#include "vtkKWIcon.h"
#include "vtkSlicerIcons.h"
#include "./Resources/vtkQueryAtlas_ImageData.h"

class vtkQueryAtlasIcons : public vtkSlicerIcons
{
 public:
    static vtkQueryAtlasIcons* New ( );
    vtkTypeRevisionMacro ( vtkQueryAtlasIcons, vtkSlicerIcons);
    void PrintSelf ( ostream& os, vtkIndent indent );

    vtkGetObjectMacro ( OntologyBrowserIcon, vtkKWIcon);
    vtkGetObjectMacro ( OntologyBrowserDisabledIcon, vtkKWIcon );
    vtkGetObjectMacro ( AddIcon, vtkKWIcon);
    vtkGetObjectMacro ( DeleteIcon, vtkKWIcon);
    vtkGetObjectMacro ( UseAllIcon, vtkKWIcon );
    vtkGetObjectMacro ( UseNoneIcon, vtkKWIcon );
    vtkGetObjectMacro ( ClearAllIcon, vtkKWIcon );
    vtkGetObjectMacro ( ClearSelectedIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAnyIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAllIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithExactIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAnySelectedIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAllSelectedIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithExactSelectedIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAnyDisabledIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithAllDisabledIcon, vtkKWIcon );    
    vtkGetObjectMacro ( WithExactDisabledIcon, vtkKWIcon );    
    vtkGetObjectMacro ( SearchIcon, vtkKWIcon );
    vtkGetObjectMacro ( ReserveURIsIcon, vtkKWIcon );
    vtkGetObjectMacro ( ReserveSelectedURIsIcon, vtkKWIcon );
    
    void AssignImageDataToIcons ( );
    
 protected:
    vtkQueryAtlasIcons ( );
    ~vtkQueryAtlasIcons ( );
    vtkKWIcon *OntologyBrowserIcon;
    vtkKWIcon *OntologyBrowserDisabledIcon;
    vtkKWIcon *AddIcon;
    vtkKWIcon *DeleteIcon;
    vtkKWIcon *UseAllIcon;
    vtkKWIcon *UseNoneIcon;
    vtkKWIcon *ClearSelectedIcon;
    vtkKWIcon *ClearAllIcon;
    vtkKWIcon *WithAnyIcon;
    vtkKWIcon *WithAllIcon;
    vtkKWIcon *WithExactIcon;    
    vtkKWIcon *WithAnySelectedIcon;
    vtkKWIcon *WithAllSelectedIcon;
    vtkKWIcon *WithExactSelectedIcon;    
    vtkKWIcon *WithAnyDisabledIcon;
    vtkKWIcon *WithAllDisabledIcon;
    vtkKWIcon *WithExactDisabledIcon;    
    vtkKWIcon *SearchIcon;
    vtkKWIcon *ReserveURIsIcon;
    vtkKWIcon *ReserveSelectedURIsIcon;
    
 private:
    vtkQueryAtlasIcons (const vtkQueryAtlasIcons&); // Not implemented
    void operator = ( const vtkQueryAtlasIcons& ); // Not implemented
    
};

#endif
