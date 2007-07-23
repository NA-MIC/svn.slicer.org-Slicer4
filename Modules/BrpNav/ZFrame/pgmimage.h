/*============================================================*
 *                                                            *
 * Simple PGM Image Format Reader/Writer                      *
 *                                                            *
 * Created by Junichi Tokuda.                                 *
 *                                                            *
 *============================================================*
 * Description: PGM image format                              *
 *============================================================*/

#include <iostream>

#ifndef _INC_PGM_IMAGE
#define _INC_PGM_IMAGE

class PgmImage {

private:
  int   width;
  int   height;

  unsigned char* image;
  short* simage;

public:

  PgmImage();
  PgmImage(int, int, unsigned char*);
  PgmImage(int, int, short*);
  ~PgmImage();

  void getSize(int* w, int* h);

  inline unsigned char* getDataC() {
    return image;
  };

  inline short* getDataS() {
    return simage;
  };
  
  inline unsigned char  getPixel(int x, int y) {
    return (x <= width && y <= height)? image[y*width + x] : 0;
  };

  inline bool isRead() {
    return (image != NULL)? true : false;
  };

private:
  void char2short();
  void short2char();

  friend std::ostream& operator<<(std::ostream& os, const PgmImage& object);
  friend std::istream& operator>>(std::istream& is, PgmImage& object);
  
};

std::ostream& operator<<(std::ostream& os, const PgmImage& object);
std::istream& operator>>(std::istream& is, PgmImage& object);

#endif // _INC_PGM_IMAGE
