#include "vtkITKBSplineTransform.h"

#include "vtkObjectFactory.h"

// Helper classes to handle dynamic setting of spline orders
class vtkITKBSplineTransformHelper
{
public:
  typedef itk::Array< double > ParametersType;
  virtual unsigned GetOrder() = 0;
  virtual unsigned int GetNumberOfParameters() = 0;
  virtual void SetParameters( ParametersType const& ) = 0;
  virtual void SetParameters( vtkDoubleArray& param ) = 0;
  virtual void SetParameters( double const* ) = 0;
  
  virtual ParametersType const& GetParameters() const = 0;

  virtual void SetFixedParameters( double const*, unsigned N ) = 0;
  virtual const double* GetFixedParameters( unsigned& N ) const = 0;

  virtual void SetGridOrigin( const double origin[3] ) = 0;
  virtual void SetGridSpacing( const double spacing[3] ) = 0;
  virtual void SetGridSize( const unsigned int size[3] ) = 0;
  virtual void ForwardTransformPoint( const double in[3], double out[3] ) = 0;
  virtual void ForwardTransformPoint( const float in[3], float out[3] ) = 0;
  virtual void ForwardTransformDerivative( const double in[3], double out[3],
                                           double derivative[3][3] ) = 0;
  virtual void ForwardTransformDerivative( const float in[3], float out[3],
                                           float derivative[3][3] ) = 0;
  virtual void InverseTransformDerivative( const double in[3], double out[3],
                                           double derivative[3][3] )=0;
  virtual void InverseTransformDerivative( const float in[3], float out[3],
                                           float derivative[3][3] )=0;
  virtual void InverseTransformPoint( const float in[3], float out[3] ) = 0;
  virtual void InverseTransformPoint( const double in[3], double out[3] ) = 0;
};


template<unsigned O>
class vtkITKBSplineTransformHelperImpl : public vtkITKBSplineTransformHelper
{
public:
  typedef itk::BSplineDeformableTransform< double, 3, O > BSplineType;

  vtkITKBSplineTransformHelperImpl():BSpline( BSplineType::New() ) {}
  virtual unsigned GetOrder() { return O; }
  virtual unsigned int GetNumberOfParameters() 
  { return BSpline->GetNumberOfParameters(); }
  virtual ParametersType const& GetParameters() const
  { return BSpline->GetParameters(); }
  virtual void SetParameters( ParametersType const& );
  virtual void SetParameters( vtkDoubleArray& param );
  virtual void SetParameters( double const* );
  virtual void SetFixedParameters( double const*, unsigned N );
  virtual const double* GetFixedParameters( unsigned& N ) const;
  virtual void SetGridOrigin( const double origin[3] );
  virtual void SetGridSpacing( const double spacing[3] );
  virtual void SetGridSize( const unsigned int size[3] );
  virtual void ForwardTransformPoint( const double in[3], double out[3] );
  virtual void ForwardTransformPoint( const float in[3], float out[3] );
  virtual void ForwardTransformDerivative( const double in[3], double out[3],
                                           double derivative[3][3] );
  virtual void ForwardTransformDerivative( const float in[3], float out[3],
                                           float derivative[3][3] );
  virtual void InverseTransformDerivative( const double in[3], double out[3],
                                           double derivative[3][3] );
  virtual void InverseTransformDerivative( const float in[3], float out[3],
                                           float derivative[3][3] );
  virtual void InverseTransformPoint( const float in[3], float out[3] );
  virtual void InverseTransformPoint( const double in[3], double out[3] );
private:
  typename BSplineType::Pointer BSpline;
  typename BSplineType::ParametersType parameters;
};


//---------------------------------------------------------------------------
// Implementation of main class.
// This mostly just forwards everything to the helper class.


vtkCxxRevisionMacro(vtkITKBSplineTransform, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkITKBSplineTransform);

void
vtkITKBSplineTransform
::PrintSelf( ostream& os, vtkIndent indent )
{
  this->Superclass::PrintSelf(os,indent);

  if( Helper != NULL )
    {
    os << indent << "Spline order: " << Helper->GetOrder() << "\n";
    os << indent << "Num parameters: " << Helper->GetNumberOfParameters() << "\n";
    }
  else
    {
    os << indent << "(no spline)\n";
    }
}


vtkAbstractTransform*
vtkITKBSplineTransform
::MakeTransform()
{
  vtkITKBSplineTransform* N = new vtkITKBSplineTransform;
  if( Helper ) 
  {
    N->SetSplineOrder( Helper->GetOrder() );
    N->Helper->SetParameters( Helper->GetParameters() );
  }
  return N;
}

vtkITKBSplineTransform
::vtkITKBSplineTransform() :
  Helper( 0 )
{
}

vtkITKBSplineTransform
::~vtkITKBSplineTransform()
{
  delete Helper;
}


void
vtkITKBSplineTransform
::SetSplineOrder( unsigned int order )
{
  if( Helper && Helper->GetOrder() == order )
  {
    return;
  }

  delete Helper;
  switch( order ) 
  {
  case 2:
    Helper = new vtkITKBSplineTransformHelperImpl< 2 >;
    break;
  case 3:
    Helper = new vtkITKBSplineTransformHelperImpl< 3 >;
    break;
  default:
    vtkErrorMacro( "order " << order << " not yet implemented" );
    break;
  }
}


unsigned int
vtkITKBSplineTransform
::GetSplineOrder() const
{
  if( Helper )
  {
    return Helper->GetOrder();
  }
  else
  {
    return 0;
  }
}


void 
vtkITKBSplineTransform
::SetGridOrigin( const double origin[3] ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetGridOrigin( origin );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetGridOrigin" );
    }
}

void 
vtkITKBSplineTransform
::SetGridSpacing( const double spacing[3] ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetGridSpacing( spacing );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetGridSpacing" );
    }
}

void
vtkITKBSplineTransform
::SetGridSize( const unsigned int size[3] ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetGridSize( size );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetGridSize" );
    }
}

void
vtkITKBSplineTransform
::SetParameters( vtkDoubleArray& param ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetParameters( param );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetParameters" );
    }
}

void
vtkITKBSplineTransform
::SetParameters( double const* param ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetParameters( param );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetParameters" );
    }
}

unsigned int 
vtkITKBSplineTransform
::GetNumberOfParameters() const 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    return Helper->GetNumberOfParameters();
    }
  else
    {
    return 0;
    }
}

double const*
vtkITKBSplineTransform
::GetParameters() const 
{
  if( Helper != NULL )
    {
    return Helper->GetParameters().data_block();
    }
  else
    {
    return NULL;
    }
}

void
vtkITKBSplineTransform
::SetFixedParameters( double const* param, unsigned N ) 
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->SetFixedParameters( param, N );
    }
  else
    {
    vtkErrorMacro( "need to call SetSplineOrder before SetFixedParameters" );
    }
}

unsigned int
vtkITKBSplineTransform
::GetNumberOfFixedParameters() const
{
  unsigned N = 0;
  if( Helper != NULL )
    {
    Helper->GetFixedParameters( N );
    }
  return N;
}

const double*
vtkITKBSplineTransform
::GetFixedParameters() const
{
  if( Helper != NULL )
    {
    unsigned N;
    return Helper->GetFixedParameters(N);
    }
  else
    {
    return NULL;
    }
}

void
vtkITKBSplineTransform
::ForwardTransformPoint( const float in[3], float out[3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->ForwardTransformPoint( in, out );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    }
}

void 
vtkITKBSplineTransform
::ForwardTransformPoint( const double in[3], double out[3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->ForwardTransformPoint( in, out );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    }
}

void
vtkITKBSplineTransform
::ForwardTransformDerivative( const float in[3], float out[3],
                              float derivative[3][3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->ForwardTransformDerivative( in, out, derivative );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    for( unsigned i = 0; i < 3; ++i )
      for( unsigned j = 0; j < 3; ++j )
        {
          derivative[i][j] = i==j ? 1.0f : 0.0f;
        }
    }
}

void
vtkITKBSplineTransform
::ForwardTransformDerivative( const double in[3], double out[3],
                              double derivative[3][3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->ForwardTransformDerivative( in, out, derivative );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    for( unsigned i = 0; i < 3; ++i )
      for( unsigned j = 0; j < 3; ++j )
        {
          derivative[i][j] = i==j ? 1.0 : 0.0;
        }
    }
}

void
vtkITKBSplineTransform
::InverseTransformPoint( const float in[3], float out[3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->InverseTransformPoint( in, out );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    }
}

void
vtkITKBSplineTransform
::InverseTransformPoint( const double in[3], double out[3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->InverseTransformPoint( in, out );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    }
}

void
vtkITKBSplineTransform
::InverseTransformDerivative( const float in[3], float out[3],
                              float derivative[3][3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->InverseTransformDerivative( in, out, derivative );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    for( unsigned i = 0; i < 3; ++i )
      for( unsigned j = 0; j < 3; ++j )
        {
          derivative[i][j] = i==j ? 1.0f : 0.0f;
        }
    }
}

void
vtkITKBSplineTransform
::InverseTransformDerivative( const double in[3], double out[3],
                              double derivative[3][3] )
{
  // Need to have called SetSplineOrder before calling this.
  if( Helper != NULL )
    {
    Helper->InverseTransformDerivative( in, out, derivative );
    }
  else
    {
    for( unsigned i = 0; i < 3; ++i )
      {
      out[i] = in[i];
      }
    for( unsigned i = 0; i < 3; ++i )
      for( unsigned j = 0; j < 3; ++j )
        {
          derivative[i][j] = i==j ? 1.0 : 0.0;
        }
    }
}


// ---------------------------------------------------------------------------
// implement the actual wrapper around the itkBSplineDeformableTransform


template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::SetGridOrigin( const double vOrigin[3] )
{
  typename BSplineType::OriginType origin;
  for( unsigned int i=0; i<3; ++i )
    origin[i] = vOrigin[i];
  BSpline->SetGridOrigin( origin );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::SetGridSpacing( const double vSpacing[3] )
{
  typename BSplineType::SpacingType spacing;
  for( unsigned int i=0; i<3; ++i )
    spacing[i] = vSpacing[i];
  BSpline->SetGridSpacing( spacing );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::SetGridSize( const unsigned int vSize[3] )
{
  typename BSplineType::RegionType region;
  typename BSplineType::RegionType::IndexType index;
  typename BSplineType::RegionType::SizeType size;
  for( unsigned int i=0; i<3; ++i ) {
    index[i] = 0;
    size[i] = vSize[i];
  }
  region.SetSize( size );
  region.SetIndex( index );
  BSpline->SetGridRegion( region );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::SetParameters( ParametersType const& param )
{
  BSpline->SetParameters( param );
}

template< unsigned O >
void 
vtkITKBSplineTransformHelperImpl<O>
::SetParameters( vtkDoubleArray& param )
{
  unsigned numberOfParam = BSpline->GetNumberOfParameters();
  this->parameters.SetSize( numberOfParam );
                                         
  for( unsigned int i=0; i<numberOfParam; ++i )
    this->parameters.SetElement( i, param.GetTuple1(i) );

  BSpline->SetParameters( parameters );
}

template< unsigned O >
void 
vtkITKBSplineTransformHelperImpl<O>
::SetParameters( double const* param )
{
  unsigned numberOfParam = BSpline->GetNumberOfParameters();
  this->parameters.SetSize( numberOfParam );
                                         
  for( unsigned int i=0; i<numberOfParam; ++i )
    this->parameters.SetElement( i, param[i] );

  BSpline->SetParameters( parameters );
}

template< unsigned O >
void 
vtkITKBSplineTransformHelperImpl<O>
::SetFixedParameters( double const* param, unsigned N )
{
  typename BSplineType::ParametersType parameters( N );
                                         
  for( unsigned int i=0; i<N; ++i )
    parameters.SetElement( i, param[i] );

  BSpline->SetFixedParameters( parameters );
}


template< unsigned O >
const double*
vtkITKBSplineTransformHelperImpl<O>
::GetFixedParameters( unsigned& N ) const
{
  N = BSpline->GetFixedParameters().GetSize();
  return BSpline->GetFixedParameters().data_block();
}


template <class T, unsigned O>
void
ForwardTransformHelper( typename itk::BSplineDeformableTransform<double, 3, O>::Pointer BSpline, 
                        const T in[3], T out[3] )
{
  typedef itk::BSplineDeformableTransform<double, 3, O> BSplineType;
  typename BSplineType::InputPointType inputPoint;

  inputPoint[0] = in[0];
  inputPoint[1] = in[1]; 
  inputPoint[2] = in[2];

  typename BSplineType::OutputPointType outputPoint;
  outputPoint = BSpline->TransformPoint( inputPoint );

  out[0] = static_cast<T>(outputPoint[0]); 
  out[1] = static_cast<T>(outputPoint[1]); 
  out[2] = static_cast<T>(outputPoint[2]);
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::ForwardTransformPoint( const double in[3], double out[3] )
{
  ForwardTransformHelper<double, O>( BSpline, in, out );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::ForwardTransformPoint( const float in[3], float out[3] )
{
  ForwardTransformHelper<float, O>( BSpline, in, out );
}

template <class T, unsigned O>
void
ForwardTransformDerivativeHelper( typename itk::BSplineDeformableTransform<double, 3, O>::Pointer BSpline, 
                                  const T in[3], T out[3], 
                                  T derivative[3][3] )
{
  typedef itk::BSplineDeformableTransform<double, 3, O> BSplineType;
  typename BSplineType::InputPointType inputPoint;

  inputPoint[0] = in[0];
  inputPoint[1] = in[1]; 
  inputPoint[2] = in[2];

  typename BSplineType::OutputPointType outputPoint;
  outputPoint = BSpline->TransformPoint( inputPoint );

  out[0] = static_cast<T>( outputPoint[0] ); 
  out[1] = static_cast<T>( outputPoint[1] ); 
  out[2] = static_cast<T>( outputPoint[2] );

  typename BSplineType::JacobianType jacobian = BSpline->GetJacobian( inputPoint );
  for( unsigned i=0; i<3; ++i )
  {
    derivative[i][0] = static_cast<T>( jacobian( i, 0 ) );
    derivative[i][1] = static_cast<T>( jacobian( i, 1 ) );
    derivative[i][2] = static_cast<T>( jacobian( i, 2 ) );
  }
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::ForwardTransformDerivative( const double in[3], double out[3],
                            double derivative[3][3] )
{
  ForwardTransformDerivativeHelper<double, O>( BSpline, in, out, derivative );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::ForwardTransformDerivative( const float in[3], float out[3],
                            float derivative[3][3] )
{
  ForwardTransformDerivativeHelper<float, O>( BSpline, in, out, derivative );
}


template <class T, unsigned O>
void
InverseTransformPointHelper( typename itk::BSplineDeformableTransform<double, 3, O>::Pointer BSpline, 
                             const T in[3], T out[3] )
{
  typedef itk::BSplineDeformableTransform<double, 3, O> BSplineType;
  typedef typename BSplineType::OutputPointType OutputPointType;
  typedef typename BSplineType::InputPointType InputPointType;

  //iterative inverse bSpline transform
  int const MaxIterationNumber = 10;
  double const Tolerance = 1;
  OutputPointType opt;
  InputPointType ipt;
  opt[0] = in[0];
  opt[1] = in[1];
  opt[2] = in[2];
  ipt[0] = opt[0];
  ipt[1] = opt[1];
  ipt[2] = opt[2];
  for (int k = 0; k <= MaxIterationNumber; k++)
  {
    OutputPointType optTrail = BSpline->TransformPoint( ipt );
    ipt[0] = ipt[0] + (opt[0]-optTrail[0]);
    ipt[1] = ipt[1] + (opt[1]-optTrail[1]);
    ipt[2] = ipt[2] + (opt[2]-optTrail[2]);
    double dist = fabs(opt[0]-optTrail[0])+fabs(opt[1]-optTrail[1])+fabs(opt[2]-optTrail[2]);
    if (dist < Tolerance )
    {
      break;
    }
  }

  out[0] = static_cast<T>(ipt[0]); 
  out[1] = static_cast<T>(ipt[1]); 
  out[2] = static_cast<T>(ipt[2]);
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::InverseTransformPoint( const float in[3], float out[3] )
{
  InverseTransformPointHelper<float, O>( BSpline, in, out );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::InverseTransformPoint( const double in[3], double out[3] )
{
  InverseTransformPointHelper<double, O>( BSpline, in, out );
}


template <class T, unsigned O>
void
InverseTransformDerivativeHelper( typename itk::BSplineDeformableTransform<double, 3, O>::Pointer BSpline, 
                                  const T in[3], T out[3], T derivative[3][3] )
{
  typedef itk::BSplineDeformableTransform<double, 3, O> BSplineType;
  typedef typename BSplineType::InputPointType InputPointType;
  typedef typename BSplineType::JacobianType JacobianType;

  InverseTransformPointHelper<T,O>( BSpline, in, out );

  InputPointType pt;

  pt[0] = out[0];
  pt[1] = out[1]; 
  pt[2] = out[2];

  JacobianType const& jacobian = BSpline->GetJacobian( pt );
  for( unsigned i=0; i<3; ++i )
  {
    derivative[i][0] = static_cast<T>( jacobian( i, 0 ) );
    derivative[i][1] = static_cast<T>( jacobian( i, 1 ) );
    derivative[i][2] = static_cast<T>( jacobian( i, 2 ) );
  }  
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::InverseTransformDerivative( const float in[3], float out[3],
                            float derivative[3][3] )
{
  InverseTransformDerivativeHelper<float, O>( BSpline, in, out, derivative );
}

template< unsigned O >
void
vtkITKBSplineTransformHelperImpl<O>
::InverseTransformDerivative( const double in[3], double out[3],
                            double derivative[3][3] )
{
  InverseTransformDerivativeHelper<double, O>( BSpline, in, out, derivative );
}
