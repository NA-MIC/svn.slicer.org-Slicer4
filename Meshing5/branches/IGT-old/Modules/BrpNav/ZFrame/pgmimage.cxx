/*============================================================*
 *                                                            *
 * Simple PGM Image Format Reader/Writer                      *
 *                                                            *
 * Created by Junichi Tokuda.                                 *
 *                                                            *
 *============================================================*
 * Description: PGM image format                              *
 *============================================================*/

#include <sstream>
#include <string.h>
#include "pgmimage.h"

PgmImage::PgmImage()
{
  image  = NULL;
  simage = NULL;
  width   = 0;
  height  = 0;
}


PgmImage::PgmImage(int w, int h, unsigned char* data)
{
  width  = w;
  height = h;
  image  = new unsigned char[w*h];
  simage = new short[w*h];

  memcpy(image, data, w*h*sizeof(unsigned char));
  char2short();
}


PgmImage::PgmImage(int w, int h, short* data)
{
  width  = w;
  height = h;
  image  = new unsigned char[w*h];
  simage = new short[w*h];

  memcpy(simage, data, w*h*sizeof(short));
  short2char();
}


PgmImage::~PgmImage()
{
  if (image)
    delete [] image;
  if (simage)
    delete [] simage;
}


void PgmImage::getSize(int* w, int* h)
{
  *w = this->width;
  *h = this->height;
}


void PgmImage::char2short()
{
  if (image && simage) {
    unsigned char*  pend = image + width*height;
    unsigned char*  pc   = image;
    short*          ps   = simage;

    while (pc < pend) {
      *ps = (short) *pc;
      pc ++;
      ps ++; 
    }
  }
}


void PgmImage::short2char()
{
  if (image && simage) {
    unsigned char* pend = image + width*height;
    unsigned char* pc  = image;
    short*         ps  = simage;

    while (pc < pend) {
      *pc = (unsigned char)*ps;
      ps ++; 
      pc ++;
    }
  }
}


std::istream& operator>>(std::istream& is, PgmImage& object)
{
  std::string buf;
  int w, h, d;

  getline(is, buf);
  if (buf != "P5") {
    std::cerr << "Not supported data!" << std::endl;
    if (object.image)
      delete [] object.image;
    object.width = object.height = 0;
  }

  while (getline(is, buf) && buf.at(0) == '#') {
    // skip comments
  }

  // get width and height
  std::istringstream iss(buf);
  iss >> w;
  iss >> h;

  // get depth
  is  >> d;
  getline(is,buf);

  // check parameters
  if (object.image) {
    delete [] object.image;
  }
  if (object.simage) {
    delete [] object.simage;
  }
  object.image  = new unsigned char[w*h];
  object.simage = new short[w*h];

  // load binary data
  if (object.image) {
    object.width  = w;
    object.height = h;
    //is >> object.image;
    is.read((char*)object.image, w*h);
    object.char2short();
  } else {
    object.width  = 0;
    object.height = 0;
  }

  return is;

}


std::ostream& operator<<(std::ostream& os, const PgmImage& object)
{

  // magic number
  os << "P5" << std::endl;
  os << "# CREATOR: PgmImage class" << std::endl;
  os << object.width << " " << object.height << std::endl;
  os << "256" << std::endl;

  os.write((char*)object.image, object.width*object.height);
    
  return os;
}







