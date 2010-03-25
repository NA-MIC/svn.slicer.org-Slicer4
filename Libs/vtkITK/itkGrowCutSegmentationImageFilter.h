#ifndef __itkGrowCutSegmentationImageFilter_h
#define __itkGrowCutSegmentationImageFilter_h
#include "itkImage.h"
#include "itkImageToImageFilter.h"
#include "itkSimpleDataObjectDecorator.h"
#include "itkVectorContainer.h"
#include "vcl_list.h"

namespace itk
{

/**  /class GrowCutSegmentationFilter
 *   \brief Given a mask image containing the gestures for foreground classes and their background, employs 
 *   grow cut segmentation to produce segmentation. Supports passing an existing segmentation with additional 
 *   gestures for editing a segmentation produced as a result of the same or different algorithm.
 *
 *    The filter is based on the paper "GrowCut:Interactive Multi-Label 
 *    N-D Image Segmentation By Cellular Automata", Vladimir Vezhnevets, 
 *    Vadim Konouchine
 *
 *    Modified Version: The inputs consist of the input intensity image, 
 *    optionally the input Label Image, and the weight image. 
 *    The Label image and the weight image default to using unsigned chars and 
 *    double respectively.
 *
 *    This algorithm is implemented scalar images. Vector Images 
 *    are not supported.
 *    
 *        
**/


/* template<class TInputImage,  */
/*   class TOutputImage, class TLabelPixelType = short,  */
/*   class TWeightPixelType = float >  */
template<class TInputImage, 
  class TOutputImage, 
  class TWeightPixelType = float> 
  class ITK_EXPORT GrowCutSegmentationImageFilter: public ImageToImageFilter<TInputImage,TOutputImage> 
{

 public:
  /** Standard class typedefs. */
  typedef GrowCutSegmentationImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods).  */
  itkTypeMacro(GrowCutSegmentationImageFilter,
               ImageToImageFilter);
 
  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension ) ;

  typedef TInputImage InputImageType;
  typedef typename InputImageType::Pointer InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType InputImageRegionType; 
  typedef typename InputImageType::PixelType InputImagePixelType; 
  //typedef typename InputImageType::IndexType InputIndexType;
  typedef typename InputImageType::SizeType SizeType;
  
  typedef TOutputImage OutputImageType;
  typedef typename OutputImageType::Pointer OutputImagePointer;
  typedef typename OutputImageType::RegionType OutputImageRegionType; 
  typedef typename OutputImageType::PixelType OutputImagePixelType;

  /** Smart Pointer type to a DataObject. */
  typedef typename DataObject::Pointer DataObjectPointer;

   /** Image dimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

   /** Index typedef support. */
  typedef Index<itkGetStaticConstMacro(InputImageDimension)> IndexType;
  
 /** InputSizeType typedef support **/
  typedef typename InputImageType::SizeType InputSizeType;

  /* NodeContainer typedef support for storing a set of seed points */
  typedef VectorContainer<unsigned int, IndexType> NodeContainer;
  
  /* NodeContainer pointer support */
  typedef typename NodeContainer::Pointer NodeContainerPointer;

  /** enum of the growcut segmentation. NoLabel represents points 
    that have not been assigned any label. Object represents points 
    that are assigned to foreground, and Background represents points 
    that are assigned to the background **/
  enum LabelType { NoLabel, ObjectLabel, BackgroundLabel };

    /** LabelImage typedef support. */  
/*    typedef Image<unsigned char, itkGetStaticConstMacro(InputImageDimension)> LabelImageType;  */

    /** LabelImagePointer typedef support. */  
/*    typedef typename LabelImageType::Pointer LabelImagePointer;  */
/*    typedef typename LabelImageType::PixelType LabelImagePixelType;   */

  
  /** StrengthImage typedef support */
  /* indicates the strength of a label for a given cell */
  //typedef Image<double, itkGetStaticConstMacro(InputImageDimension) > StrengthImageType;
   typedef Image<TWeightPixelType, itkGetStaticConstMacro(InputImageDimension) > StrengthImageType;
  

  /** StrengthImagePointer typedef support. */
  typedef typename StrengthImageType::Pointer StrengthImagePointer;
  
 typedef TWeightPixelType StrengthImagePixelType;
 //typedef typename StrengthImageType::PixelType StrengthImagePixelType;

 typedef OutputImageType LabelImageType;

 typedef typename LabelImageType::Pointer LabelImagePointer;
 
 typedef typename LabelImageType::PixelType LabelImagePixelType;

  // set the foreground seeds
  void SetForegroundPoints( NodeContainer *points )
  {
    m_ForegroundPoints = points;
    this->Modified();
  }

  // set the background seeds
  void SetBackgroundPoints( NodeContainer *points )
  {
    m_BackgroundPoints = points;
    this->Modified();
  }

  // set the prior foreground points
  void SetPriorForegroundPoints( NodeContainer *points )
  {
    m_PriorForegroundPoints = points;
    this->Modified();
  }

  /** Get the container of foreground seeds **/
  NodeContainerPointer GetForegroundPoints()
  {
    return m_ForegroundPoints;
  }

  /** Get the container of background seeds **/
  NodeContainerPointer GetBackgroundPoints()
  {
    return m_BackgroundPoints;
  }

  /** Get the container of prior foreground seeds **/
  NodeContainerPointer GetPriorForegroundPoints()
  {
    return m_PriorForegroundPoints;
  }

 /** Set the Input Image **/
 void SetInputImage( const InputImageType *in)
 {
   this->ProcessObject::SetNthInput(0, const_cast< InputImageType *>(in) );
 }

 InputImagePointer GetInputImage( ) const 
 {
   return static_cast< InputImageType *>(this->ProcessObject::GetInput(0));
 }

 /** Set the Label Image **/
 void SetLabelImage( const LabelImageType *f)
 {
    this->ProcessObject::SetNthInput(1, const_cast< LabelImageType *>(f) );
    m_LabelImage = static_cast< LabelImageType *>(this->ProcessObject::GetInput(1));
 }

  /** Get the Label image **/
  LabelImagePointer GetLabelImage() const
  {
    return m_LabelImage;
  }
  
  /** Get the Strength image **/
  StrengthImagePointer GetStrengthImage() const
  {
    return m_StrengthImage;
  }
 
 /** Set the Weight Image **/
 void SetStrengthImage( const StrengthImageType *w )
 {
   this->ProcessObject::SetNthInput(2,const_cast< StrengthImageType *>(w));
   m_PrevStrengthImage = static_cast< StrengthImageType *>(this->ProcessObject::GetInput(2));
 }

  /** Set the initial strength **/ 
  itkSetMacro( SeedStrength, double );
  
  /** Get the seed strength **/
  itkGetConstMacro( SeedStrength, double );

  /** Set the prior segmented pix strength **/ 
  itkSetMacro( PriorSegmentStrength, double );
  
  /** Get the seed strength **/
  itkGetConstMacro( PriorSegmentStrength, double );

  /** Set the number of iterations **/
  itkSetMacro( MaxIterations, unsigned int );
  
  /** Get the number of iterations **/
  itkGetConstMacro( MaxIterations, unsigned int );

  /** Set T1: Max number of enemies for a cell to continue attack **/
  itkSetMacro( T1, unsigned int );
  
  /** Get T1 **/
  itkGetConstMacro( T1, unsigned int );

  /** Set T2: Min number of enemies for a cell to be consumed **/
  itkSetMacro( T2, unsigned int );
  
  /** Get T2 **/
  itkGetConstMacro( T2, unsigned int );

  /** Set Algorithm Speed Type: Slow or Optimized neighborhood **/
  itkSetMacro( UseSlow, bool);
  
  /** Get the neighborhood used **/
  itkGetConstMacro( UseSlow, bool );

 /** Set the radius of the neighborhood for processing. Default is 1 **/
 itkSetMacro( Radius, InputSizeType );
  
 /** Get the radius of neighborhood used for processing **/
 itkGetConstMacro( Radius, InputSizeType );



 protected:
  
  GrowCutSegmentationImageFilter();
  ~GrowCutSegmentationImageFilter() {};

   // Override since the filter needs all the data for the algorithm
  void GenerateInputRequestedRegion();

   // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

  void GenerateData();  
  
  void Initialize( OutputImageType *);
  
  void PrintSelf ( std::ostream& os, Indent indent ) const;

  void GrowCutSlow( OutputImageType *);
  
 // void GrowCutFast( OutputImageType *);

  
 private:

  GrowCutSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); // purposely not implemented


  NodeContainerPointer                       m_ForegroundPoints; 
  NodeContainerPointer                       m_BackgroundPoints;
  NodeContainerPointer                       m_PriorForegroundPoints;

  InputSizeType                              m_Radius; 
 
  LabelImagePointer                          m_LabelImage;
  
  StrengthImagePointer                       m_StrengthImage;

  StrengthImagePointer                       m_PrevStrengthImage;

  OutputImageRegionType                      m_BufferedRegion;

  double                                     m_SeedStrength;
 
  double                                     m_PriorSegmentStrength;

  unsigned int                               m_MaxIterations;
  unsigned int                               m_T1; // max number of enemies a cell can have to attack
  unsigned int                               m_T2; // min number of enemies a cell can have to be consumed
  
  bool                                       m_UseSlow;

 OutputImagePixelType                        m_ObjectLabel;
 OutputImagePixelType                        m_BackgroundLabel;
 OutputImagePixelType                        m_UnknownLabel;

 vcl_list< IndexType >                      m_AlivePoints;   
  

};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGrowCutSegmentationImageFilter.txx"
#endif








#endif
