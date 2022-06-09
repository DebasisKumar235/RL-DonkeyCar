import os
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import cv2

from ae.autoencoder import load_ae

max_num_stacked_images = 10
real_amount = 2

def crop_and_reverse( img ):
        CAMERA_HEIGHT = 120
        CAMERA_WIDTH = 160

        MARGIN_TOP = CAMERA_HEIGHT // 3

        # Region Of Interest
        # r = [margin_left, margin_top, width, height]
        ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]
        r = ROI
        return img[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    

class AutoencoderWrapper(gym.Wrapper):
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param ae_path: Path to the autoencoder
    """

    def __init__(self, env: gym.Env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):  # noqa: B008
        super().__init__(env)
        assert ae_path is not None, "No path to autoencoder was provided"
        self.ae = load_ae(ae_path)
        self.stack_of_images = []
        print( "My autoencoder!!!" )
        # Update observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size*real_amount,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        # Important: Convert to BGR to match OpenCV convention
        ##encoded_image = self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1])
        #new_obs = np.concatenate( [ encoded_image.flatten(), [0.0] ] )
        ##return encoded_image.flatten()
        self.stack_of_images = []
        return self.___get_observation( self.env.reset() )    

    def _local_encode( self, obs ):
        #encoded = self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten()
        cropped_img = crop_and_reverse(obs)
        cropped_img = cropped_img[:, :, ::-1]
        encoded = self.ae.encode(cropped_img)
        return encoded

    def ___get_observation( self, obs ):
        
        encoded = self._local_encode( obs )

        if len( self.stack_of_images ) == max_num_stacked_images:
            self.stack_of_images.pop(0)
     
        self.stack_of_images.append( encoded )

        if len( self.stack_of_images ) == max_num_stacked_images:
            return np.hstack( [self.stack_of_images[-1], self.stack_of_images[0]] )
        else:
            stack = [encoded] #self.stack_of_images.copy()

            dummy = np.zeros( np.asarray( encoded ).shape )

            for _ in range( real_amount - 1 ):
                stack.append( dummy )
                
            return np.hstack( stack )
            #return np.hstack( [ encoded ] * max_num_stacked_images )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        #encoded_image = self.ae.encode_from_raw_image(obs[:, :, ::-1])
        #reconstructed_image = self.ae.decode( encoded_image )[0]
        #cv2.imshow( "Original image", encoded_image[:, :, ::-1] )
        # cv2.imshow( "Reconstructed image", reconstructed_image )
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass

        #speed = infos['speed']
        #new_obs = np.concatenate( [ self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten(), [speed] ] )
        return self.___get_observation( obs ), reward, done, infos


class AutoencoderWrapper2:
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param ae_path: Path to the autoencoder
    """

    def __init__(self, ae_path: Optional[str] = os.environ.get("AAE_PATH")):  # noqa: B008
        assert ae_path is not None, "No path to autoencoder was provided"
        self.ae = load_ae(ae_path)
        self.stack_of_images = []

    def reset( self ):
        self.stack_of_images = []

    def encode_observation( self, obs ):
        
        cropped_img = crop_and_reverse(obs)
        encoded = self.ae.encode(cropped_img[:, :, ::-1])

        if len( self.stack_of_images ) == max_num_stacked_images:
            self.stack_of_images.pop(0)
     
        self.stack_of_images.append( encoded )

        if len( self.stack_of_images ) == max_num_stacked_images:
            #res = np.hstack( self.stack_of_images )
            #return np.expand_dims( res, axis=0 ), cropped_img
            #return self.stack_of_images, cropped_img
            return [self.stack_of_images[-1], self.stack_of_images[0]], cropped_img
        else:
            # stack = self.stack_of_images.copy()

            # dummies = np.zeros( np.asarray( encoded ).shape )

            # dummy = [ dummies ] * ( max_num_stacked_images - len( stack ) )
            # for d in dummy:
            #     stack.append( d )
            stack = [encoded] #self.stack_of_images.copy()

            dummy = np.zeros( np.asarray( encoded ).shape )

            for _ in range( real_amount - 1 ):
                stack.append( dummy )

            return stack, cropped_img

    def decode_encoding( self, encoded ):
        return self.ae.decode(encoded)[0]