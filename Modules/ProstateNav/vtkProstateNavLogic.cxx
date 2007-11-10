/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkProstateNavLogic.h"

vtkCxxRevisionMacro(vtkProstateNavLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkProstateNavLogic);

const int vtkProstateNavLogic::PhaseTransitionMatrix[vtkProstateNavLogic::NumPhases][vtkProstateNavLogic::NumPhases] =
  {
               /*     next workphase     */
      /*    */ /* St  Pl  Cl  Tg  Mn  Em */
      /* St */ {  1,  1,  0,  0,  0,  1  },
      /* Pl */ {  1,  1,  1,  0,  0,  1  },
      /* Cl */ {  1,  1,  1,  1,  1,  1  },
      /* Tg */ {  1,  1,  1,  1,  1,  1  },
      /* Mn */ {  1,  1,  1,  1,  1,  1  },
      /* Em */ {  1,  1,  1,  1,  1,  1  },
  };

vtkProstateNavLogic::vtkProstateNavLogic()
{
#ifndef IGSTK_OFF
    igstk::RealTimeClock::Initialize();
#endif

    CurrentPhase         = StartUp;
    PrevPhase            = StartUp;
    PhaseComplete        = false;
    PhaseTransitionCheck = true;

}


vtkProstateNavLogic::~vtkProstateNavLogic()
{

}


void vtkProstateNavLogic::PrintSelf(ostream& os, vtkIndent indent)
{
    this->vtkObject::PrintSelf(os, indent);

    os << indent << "vtkProstateNavLogic:             " << this->GetClassName() << "\n";

}


int vtkProstateNavLogic::SwitchWorkPhase(int newwp)
{
    if (IsPhaseTransitable(newwp))
    {
        PrevPhase     = CurrentPhase;
        CurrentPhase  = newwp;
        PhaseComplete = false;

        return 1;
    }
}


int vtkProstateNavLogic::IsPhaseTransitable(int nextwp)
{
    if (nextwp < 0 || nextwp > NumPhases)
    {
        return 0;
    }
    
    if (PhaseTransitionCheck == 0)
    {
        return 1;
    }

    if (PhaseComplete)
    {
        return PhaseTransitionMatrix[CurrentPhase][nextwp];
    }
    else
    {
        return PhaseTransitionMatrix[PrevPhase][nextwp];
    }
}
