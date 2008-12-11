import time
import logging
import asyncore
import string
import socket
import numpy
import nrrd
reload(nrrd)
import slicerd
reload(slicerd)
import TensorEval as tensC
reload(tensC)
import TensorEval2 as tens
reload(tens)
import TrackFiber4 as track
reload(track)
import cmpV
reload(cmpV)
import vectors as vects

vtk_types = { 2:numpy.int8, 3:numpy.uint8, 4:numpy.int16,  5:numpy.uint16,  6:numpy.int32,  7:numpy.uint32,  10:numpy.float32,  11:numpy.float64 }

numpy_sizes = { numpy.int8:1, numpy.uint8:1, numpy.int16:2,  numpy.uint16:2,  numpy.int32:4,  numpy.uint32:4,  numpy.float32:4,  numpy.float64:8 }

numpy_nrrd_names = { 'int8':'char', 'uint8':'unsigned char', 'int16':'short',  'uint16':'ushort',  'int32':'int',  'uint32':'uint',  'float32':'float',  'float64':'double' }

numpy_vtk_types = { 'int8':'2', 'uint8':'3', 'int16':'4',  'uint16':'5',  'int32':'6',  'uint32':'7',  'float32':'10',  'float64':'11' }



logging.basicConfig(level=logging.INFO, format="%(created)-15s %(msecs)d %(levelname)8s %(thread)d %(name)s %(message)s")

logger                  = logging.getLogger(__name__)
BACKLOG                 = 5
SIZE                    = 1024


class PipelineHandler(asyncore.dispatcher):
   def __init__(self, conn_sock, client_address, server):
        self.server             = server
        self.client_address     = client_address
        self.buffer             = ""

        self.is_writable        = False
        
        self.init               = False

        self.data               = ""            # data deals with string based data (non numpy data)
        self.ntype              = ""            # ntype deals with numpy stringified arrays sent remotely - store type (not data)
        self.nflag              = False         # nflag related to numpy data recv (buffer not enough large to get it in one turn) - set a reset point for data

        self.isPartial          = False
        self.nVols              = 0
        self.ldata              = ""
        self.pvol               = []

        self.isarray            = False
        self.result             = False

        self.issent             = False
        
        self.params             = nrrd.nrrd()
        self.nimage             = nrrd.nrrd()
        self.roiA               = nrrd.nrrd()
        self.roiB               = nrrd.nrrd()
        self.wm                 = nrrd.nrrd()

        self.res                = numpy.empty(0)

        asyncore.dispatcher.__init__(self, conn_sock)
        logger.debug("created handler; waiting for loop")

   def readable(self):
        return True     

   def writable(self):
        return self.is_writable 

   def pipeline(self, data):

        if self.nimage.get('pipeline')[0]=='STOCHASTIC':

           ####! swap x and z for volume coming from Slicer - do not forget tp apply the inverse before to send them back
           data = data.swapaxes(2,0)
           ####
           shpD = data.shape
           logger.info("pipeline data shape : %s:%s:%s:%s" % (shpD[0], shpD[1], shpD[2], shpD[3]))

           orgS = self.nimage.get('origin')
           org = [float(orgS[0]), float(orgS[1]), float(orgS[2])]

           G = self.nimage.get('grads')
           b = self.nimage.get('bval')
           i2r = self.nimage.get('ijk2ras')
           mu = self.nimage.get('mu')
           dims = self.nimage.get('dimensions')

           s = slicerd.slicerd()
           scene = s.ls()
           dscene = {}
           for i in range(len(scene)/3):
               dscene[scene[(i+1)*3-1]]= scene[i*3]

               logger.info("scene : %s" % dscene)

               # currently there is a bug in the GUI of slicer python - do not load if three times the same volume 
               if self.params.hasKey('roiA'):
                   if dscene.has_key(self.params.get('roiA')[0]):
                         self.roiA = s.get(int(dscene[self.params.get('roiA')[0]]))
                         roiAR = numpy.fromstring(self.roiA.getImage(), 'uint16')
                         roiAR = roiAR.reshape(shpD[2], shpD[1], shpD[0]) # because come from Slicer - will not send them back so swap them one for all
                         roiAR = roiAR.swapaxes(2,0)
                         self.roiA.setImage(roiAR)
                         logger.info("RoiA : %s:%s:%s" % (roiAR.shape[0], roiAR.shape[1], roiAR.shape[2]))

               if self.params.hasKey('roiB'):
                   if dscene.has_key(self.params.get('roiB')[0]):
                        if self.params.get('roiB')[0] != self.params.get('roiA')[0]:
                              self.roiB = s.get(int(dscene[self.params.get('roiB')[0]]))
                              roiBR = numpy.fromstring(self.roiB.getImage(), 'uint16')
                              roiBR = roiBR.reshape(shpD[2], shpD[1], shpD[0])
                              roiBR = roiBR.swapaxes(2,0)
                              self.roiB.setImage(roiBR)
                              logger.info("RoiB : %s:%s:%s" % (roiBR.shape[0], roiBR.shape[1], roiBR.shape[2]))

               if self.params.hasKey('wm'):
                   if dscene.has_key(self.params.get('wm')[0]):
                        if self.params.get('wm')[0] != self.params.get('roiA')[0]:
                              self.wm = s.get(int(dscene[self.params.get('wm')[0]]))
                              wmR = numpy.fromstring(self.wm.getImage(), 'uint16')
                              wmR = wmR.reshape(shpD[2], shpD[1], shpD[0])
                              wmR = wmR.swapaxes(2,0)
                              self.wm.setImage(wmR)
                              logger.info("WM : %s:%s:%s" % (wmR.shape[0], wmR.shape[1], wmR.shape[2]))

               logger.info("Input volumes loaded!")

               # values per default
               smoothEnabled = False
               stdDev = [1., 1. ,1.]
               radFactors = [1.5, 1.5, 1.5]

               otsuEnabled = False
               infOtsuThres = 200
               supOtsuThres = 1000

               wmEnabled = True
               infWMThres = 300
               supWMThres = 900
               artsEnabled = False
               infARTSThres = 0
               supARTSThres = 200

               tensEnabled =True
               bLine = 0

               stEnabled = True
               totalTracts = 500
               maxLength = 200
               stepSize = 0.5
               stopEnabled = True
               fa = 0.1

               cmEnabled = False
               probMode = 0

               # got from client
               # special handling for bools
               if self.params.hasKey('smoothEnabled'):
                    smoothEnabled = bool(int(self.params.get('smoothEnabled')[0]))
               if self.params.hasKey('otsuEnabled'):
                    otsuEnabled = bool(int(self.params.get('otsuEnabled')[0]))
               if self.params.hasKey('wmEnabled'):
                    wmEnabled = bool(int(self.params.get('wmEnabled')[0]))
               if self.params.hasKey('artsEnabled'):
                    artsEnabled = bool(int(self.params.get('artsEnabled')[0]))
               if self.params.hasKey('tensEnabled'):
                    tensEnabled = bool(int(self.params.get('tensEnabled')[0]))
               if self.params.hasKey('stEnabled'):
                    stEnabled = bool(int(self.params.get('stEnabled')[0]))
               if self.params.hasKey('cmEnabled'):
                    cmEnabled = bool(int(self.params.get('cmEnabled')[0]))
               if self.params.hasKey('spaceEnabled'):
                    spaceEnabled = bool(int(self.params.get('spaceEnabled')[0]))
               if self.params.hasKey('stopEnabled'):
                    stopEnabled = bool(int(self.params.get('stopEnabled')[0]))
               if self.params.hasKey('faEnabled'):
                    faEnabled = bool(int(self.params.get('faEnabled')[0]))
               if self.params.hasKey('traceEnabled'):
                    traceEnabled = bool(int(self.params.get('traceEnabled')[0]))
               if self.params.hasKey('modeEnabled'):
                    modeEnabled = bool(int(self.params.get('modeEnabled')[0]))

               # can handle normally
               if self.params.hasKey('infOtsuThres'):
                    infOtsuThres = int(self.params.get('infOtsuThres')[0])
                    logger.debug("infOtsuThres: %s" % infOtsuThres)
               if self.params.hasKey('supOtsuThres'):
                    supOtsuThres = int(self.params.get('supOtsuThres')[0])
                    logger.debug("supOtsuThres: %s" % supOtsuThres)

               if self.params.hasKey('infARTSThres'):
                    infARTSThres = int(self.params.get('infARTSThres')[0])
                    logger.debug("infARTSThres: %s" % infARTSThres)
               if self.params.hasKey('supARTSThres'):
                    supARTSThres = int(self.params.get('supARTSThres')[0])
                    logger.debug("supARTSThres: %s" % supARTSThres)

               if self.params.hasKey('infWMThres'):
                    infWMThres = int(self.params.get('infWMThres')[0])
                    logger.debug("infWMThres: %s" % infWMThres)
               if self.params.hasKey('supWMThres'):
                    supWMThres = int(self.params.get('supWMThres')[0])
                    logger.debug("supWMThres: %s" % supWMThres)

               if self.params.hasKey('bLine'):
                    bLine = int(self.params.get('bLine')[0])
                    logger.debug("bLine: %s" % bLine)
               
               if self.params.hasKey('tensMode'):
                    tensMode = self.params.get('tensMode')[0]
                    logger.debug("tensMode: %s" % tensMode)



               if self.params.hasKey('totalTracts'):
                    totalTracts = int(self.params.get('totalTracts')[0])
                    logger.debug("totalTracts: %s" % totalTracts)
               if self.params.hasKey('maxLength'):
                    maxLength = int(self.params.get('maxLength')[0])
                    logger.debug("maxLength: %s" % maxLength)
               if self.params.hasKey('stepSize'):
                    stepSize = float(self.params.get('stepSize')[0])
                    logger.debug("stepSize: %s" % stepSize)
               if self.params.hasKey('fa'):
                    fa = float(self.params.get('fa')[0])
                    logger.debug("fa: %s" % fa)

               if self.params.hasKey('probMode'):
                    probMode = self.params.get('probMode')[0]
                    logger.debug("probMode: %s" % probMode)


               ngrads = b.shape[0]
               G = G.reshape((ngrads,3))
               b = b.reshape((ngrads,1))
               i2r = i2r.reshape((4,4))
               mu = mu.reshape((4,4))

               r2i = numpy.linalg.inv(i2r)

               logger.info("Search ROI")
               roiP1 = cmpV.march0InVolume(self.roiA.getImage())

               blocksize = totalTracts
               IJKstartpoints = numpy.tile(roiP1,( blocksize, 1))

               # correctly express gradients into RAS space
               G = numpy.dot(G, mu[:3,:3].T)

               logger.info("Tensor flag : %s" % str(tensEnabled))

               if tensEnabled:
                    logger.info("Compute tensor")
                    timeS1 = time.time()
                    EV, lV, xVTensor, otsu, wm, arts = tens.EvaluateTensorS0(data, G.T, b.T, otsuEnabled, wmEnabled, artsEnabled, bLine,\
                                  infOtsuThres, supOtsuThres,\
                                  infWMThres, supWMThres, infARTSThres, supARTSThres)

                    logger.info("Compute tensor in %s sec" % str(time.time()-timeS1))

                    if faEnabled:
                         faMap = tensC.CalculateFA(lV)

                    if stEnabled:
                         logger.info("Track fibers")
                         if not stopEnabled:
                             fa = 0.0

                         timeS2 = time.time()
                         paths = track.TrackFiber40(data, vects.vectors.T, b.T, G.T, IJKstartpoints.T, r2i, i2r,\
                                  lV, EV, xVTensor, stepSize, maxLength, fa, spaceEnabled)

                         logger.info("Track fibers in %s sec" % str(time.time()-timeS2))

                         if cmEnabled:
                              cm = track.ConnectFibers(paths, maxLength, shpD)


               elif stEnabled:
                     logger.info("Track fibers without prior tensor estimation")
                     if not stopEnabled:
                          fa = 0.0

                     timeS3 = time.time()
                     paths = track.TrackFiber4(data, vects.vectors.T, b.T, G.T, IJKstartpoints.T, r2i, i2r,\
                            stepSize, maxLength, fa, spaceEnabled)

                     logger.info("Track fibers without prior tensor estimation in %s sec" % str(time.time()-timeS3))

                     if cmEnabled:
                         cm = track.ConnectFibers(paths, maxLength, shpD)
               else:
                     logger.info("No pipeline to execute!")


               logger.info("Connect tract")

               # use slicer daemon to send back results
               if tensEnabled and artsEnabled:
                  arts = arts.swapaxes(2,0)
                  s.putS(arts, dims, org, i2r, 'arts')

               if tensEnabled and otsuEnabled:
                  otsu = otsu.swapaxes(2,0)
                  s.putS(otsu, dims, org, i2r, 'otsu')

               if tensEnabled and wmEnabled:
                  wm = wm.swapaxes(2,0)
                  s.putS(wm, dims, org, i2r, 'wm')

               if tensEnabled:
                     xVTensor = xVTensor.swapaxes(2,0)
                     s.putD(xVTensor, dims, org, i2r, mu, 'tensor')

                     if faEnabled:
                          faMap = faMap.swapaxes(2,0)
                          s.putS(faMap, dims, org, i2r, 'fa')

               if cmEnabled:
                    cm = cm.swapaxes(2,0)
                    s.putS(cm, dims, org, i2r, 'cm')

               logger.debug("pipeline data shape end : %s:%s:%s:%s" %  (shpD[0], shpD[1], shpD[2], shpD[3]))

   return data 


   def set_params(self, data):
       data = string.strip(data)
       data = string.split(data)

       if len(data)==1:
            if data[0]=='data':
                    self.init = True
                    return data[0] # data (ready to get DWI)
            else:
                    return data[0] # init
       
       tmp = data[1:]
       self.params.set(data[0], tmp)
       logger.info("param id: %s" % data[0])
       logger.info("param value: %s" % self.params.get(data[0]))


   def set_data(self, data):


        if not self.nimage.hasKey('dimensions'):

            if self.ntype == 'ijk2ras' or self.ntype == 'mu' or self.ntype == 'grads' or self.ntype == 'bval':
                self.nimage.set(self.ntype, numpy.fromstring(data, 'float'))
                logger.debug("data id: %s" % self.ntype)
                logger.debug("data value: %s" % self.nimage.get(self.ntype))
                self.ntype=""
            else:
                data = string.strip(data)
                data = string.split(data)

                if len(data)==1:
                     if data[0]=='ijk2ras' or data[0]=='mu' or data[0]=='grads' or data[0]=='bval':
                           self.ntype=data[0]
                           return
                     else:
                           return data[0]

                tmp = data[1:]
                self.nimage.set(data[0], tmp)
                logger.debug("data id: %s" % data[0])
                logger.debug("data value: %s" % self.nimage.get(data[0]))

                if self.nimage.hasKey('scalar_type') and self.nimage.hasKey('dimensions'):
                     scalar_type = self.nimage.get('scalar_type')
                     dtype = vtk_types [ int(scalar_type[0]) ]
                     size = numpy_sizes [ dtype ]
                     dimensions = self.nimage.get('dimensions')
                     logger.debug("dimensions size : %s" % len(dimensions))

                     if len(dimensions) == 4:
                           size = size * int(dimensions[2]) * int(dimensions[1]) * int(dimensions[0]) * int(dimensions[3])
                     elif len(dimensions) == 2:
                            size = size * int(dimensions[1]) * int(dimensions[0])
                     else:
                            size = size * int(dimensions[2]) * int(dimensions[1]) * int(dimensions[0])

                     self.nimage.set('size', size)

        else:
            if self.nimage.get('kinds')[0]=='scalar' or self.nimage.get('kinds')[0]=='dti':
 
                scalar_type = self.nimage.get('scalar_type')
                dtype = vtk_types [ int(scalar_type[0]) ]

                dimensions = self.nimage.get('dimensions')
                logger.debug("preparing data")

                im = numpy.fromstring (data, dtype)
                if len(dimensions) == 3:
                   im = im.reshape( int(dimensions[0]), int(dimensions[1]), int(dimensions[2]) )

                if len(dimensions) == 2: # dti
                   im = im.reshape( int(dimensions[0]), int(dimensions[1]) )

                self.nimage.setImage(im)

            elif self.nimage.get('kinds')[0]=='dwi':
                scalar_type = self.nimage.get('scalar_type')
                dtype = vtk_types [ int(scalar_type[0]) ]

                dimensions = self.nimage.get('dimensions')
                logger.debug("preparing data")

                im = numpy.zeros(( int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), int(dimensions[3])), dtype)
                for i in range(int(dimensions[3])):
                     pim = numpy.fromstring (self.pvol[i], dtype)
                     pim = pim.reshape( int(dimensions[0]), int(dimensions[1]), int(dimensions[2]) )
                     im[..., i]= pim[...]

                self.nimage.setImage(im)
            else:
                logger.info("...")
      
        

   def handle_read(self):
        # handle incomings
        if  self.nimage.hasKey('size') and not self.result:
             self.isarray = True
             size = self.nimage.get('size')
             dims = self.nimage.get('dimensions')
             nvols = int(dims[3])

             sizePerVol = size/nvols

             if not self.nflag:
                  self.data = ""
                  self.nflag = True

                  self.ldata += self.recv(sizePerVol)

             if len(self.ldata)==sizePerVol:
                  self.isPartial = True
                  self.data += self.ldata
                  self.pvol.append(self.ldata)
                  self.nVols +=1
                  self.ldata = ""

             # get DWI and run pipeline 
             if self.nVols==nvols:
                  logger.debug("volume acquired!")
  
                  self.set_data(self.data)

                  self.pvol = []
                  self.isPartial = False
                  self.nVols = 0

                  self.data = ""
                  logger.info("pipeline launchned")
                  self.res = self.pipeline(self.nimage.getImage())
                  logger.debug("result shape : %s:%s:%s" % (self.res.shape[0] , self.res.shape[1] , self.res.shape[2] ))
                  logger.debug("result type : %s" % self.res.dtype)
                  logger.info("pipeline completed")
                  self.data = 'FACK'
                  self.result = True
                  logger.debug("ready for sending!")
        else:
            if not self.result:
                  self.isarray = False
                  self.data = self.recv(SIZE)
      
                  if not self.init: # first get parameters of the pipeline
                     self.set_params(self.data)
                  elif self.data:  # second get data associated to the vector image
                     self.set_data(self.data)
                  else:
                     logger.error("command unknown")
        
           else: # special case for returning to client (Slicer) - currently SlicerDaemon used
                  if not self.issent:
                      self.data = self.recv(SIZE)
                      cmd = string.strip(self.data)
                      cmd = string.split(cmd)
                      if len(cmd)!=1:
                         logger.error("command awaited!")
                      else:
                         logger.info("command : %s" % cmd[0])
                         if cmd[0]=='get':
                             logger.info("send back data!")
                             self.data = self.res.tostring() # 'PACK'
                             self.issent = True
                         else:
                             logger.error("command unknown")
                  else:
                      logger.info("closing!")
                      self.result = False

        # determine response
        if self.data and not self.isarray and not self.result:
             self.buffer += 'ACK'
             self.is_writable = True
             self.data = ""
        elif (self.data or self.ldata) and self.isarray and not self.result:
             logger.debug("acquiring array")
             if self.isPartial:
                 self.buffer += 'ACK'
                 self.is_writable = True
                 self.isPartial = False
        elif self.data and self.result:
             self.buffer = self.data 
             self.is_writable = True

             # reset
             if self.issent:
                  self.data             = ""        # set previously
                  self.ntype            = ""

                  self.init             = False

                  self.isarray          = False
                  self.nflag            = False

                  self.params           = nrrd.nrrd()
                  self.nimage           = nrrd.nrrd()
                  self.roiA             = nrrd.nrrd()
                  self.roiB             = nrrd.nrrd()
                  self.wm               = nrrd.nrrd()

                  self.res              = numpy.empty(0)

                  self.issent           = False
                  logger.debug("data size : %s" %len(self.buffer))


             logger.info("ready to handle write!")

        else:
             logger.info("got null data")

   def handle_write(self):
        if self.buffer:
             sent = self.send(self.buffer)
             self.buffer = self.buffer[sent:]
        else:
             logger.debug("nothing to send")

        if len(self.buffer) == 0:
             self.is_writable = False

   def handle_close(self):
        logger.debug("handle_close")
        logger.info("conn_closed: client_address=%s:%s" % \
               (self.client_address[0],
                self.client_address[1]))

        self.close()



class PipelineServer(asyncore.dispatcher):
    allow_reuse_address         = False
    request_queue_size          = 5
    address_family              = socket.AF_INET
    socket_type                 = socket.SOCK_STREAM

    def __init__(self, address=('localhost', 13001), handlerClass=PipelineHandler):
        self.address            = address
        self.handlerClass       = handlerClass
        asyncore.dispatcher.__init__(self)
        self.create_socket(self.address_family,
                       self.socket_type)

        if self.allow_reuse_address:
             self.set_reuse_addr()

        self.server_bind()
        self.server_activate()

    def server_bind(self):
        self.bind(self.address)
        logger.debug("bind: address=%s:%s" % (self.address[0], self.address[1]))

    def server_activate(self):
        self.listen(self.request_queue_size)
        logger.debug("listen: backlog=%d" % self.request_queue_size)

    def fileno(self):
        return self.socket.fileno()

    def serve_forever(self):
        asyncore.loop()


    def handle_accept(self):
        (conn_sock, client_address) = self.accept()
        if self.verify_request(conn_sock, client_address):
             self.process_request(conn_sock, client_address)

    def verify_request(self, conn_sock, client_address):
        return True

    def process_request(self, conn_sock, client_address):
        logger.info("conn_made: client_address=%s:%s" % \
                  (client_address[0],
                   client_address[1]))

        self.handlerClass(conn_sock, client_address, self)
 
    def handle_close(self):
        self.close()

def main():
    """
    Launch module.
    """
    logger.info("Launching pipeline server")
    
    server = PipelineServer()
    server.serve_forever()

    logger.info("Closing pipeline server")
   

if __name__ == '__main__':
    main()
