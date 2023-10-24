from fltk import Fl
import pydart2 as pydart
import torch
from control.pitching_ball_mound_movingcam.pitching01_ppo_mp import PPO
# from ppo_mp import PPO
from PyCommon.modules.GUI import DartViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

from SkateUtils.DartMotionEdit import DartSkelMotion
from SkateUtils.Dart2bvh import dart2bvh

def main():
    MOTION_ONLY = False

    # MOTION_ONLY = True
    np.set_printoptions(precision=5)

    pydart.init()
    env_name = 'pitching01'
    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        # ppo.LoadModel('pitching01_model_202309051448/' + 'max.pt')

        # ppo.LoadModel('pitching01_model_202308272203/' + 'max.pt')
        ppo.LoadModel('pitching_ball_mound_movingcam_max.pt')
        pass

    print("num dof: ", ppo.env.skel.ndofs)
    # for joint in ppo.env.skel.joints:
    #     print(joint, joint.position_in_world_frame())


    ppo.env.flag_rsi(False)
    ppo.env.reset()
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))

    # print("motion length:", ppo.env.motion_len)
    # for bvh file
    bvh_qs = []
    bvh_file_name = 'ppo_pitching01_vibe.bvh'
    skmo_file_name = 'ppo_pitching01_vibe.skmo'
    motion = DartSkelMotion()

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    rd_ext_force = [None]
    rd_ext_force_pos = [None]
    rd_traj = []
    ref_traj = []
    rd_frame_text_label = ['frame: ']
    rd_frame_text_label_pos = [(-0.95, 0.95)]
    rd_frame_text = ['0']
    rd_frame_text_pos = [(-0.9, 0.95)]
    rd_cotact_label_text = ['[lf rf lh rh]']
    rd_cotact_label_text_pos = [(-0.95, 0.9)]

    rd_contact_inf_text = ['']
    rd_contact_inf_text_pos = [(-0.95, 0.85)]

    rd_target_zone = [None]
    rd_ball_traj = []

    render_color = []
    ##
    left_hand_size = []
    left_hand_T = []

    left_bicep_size = []
    left_bicep_T = []

    left_forearm_size = []
    left_forearm_T = []
    dart_world = ppo.env.world
    dartSkelRenderer = yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL)
    
    viewer_w, viewer_h = 1920, 1080
    viewer = hsv.DartViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55))
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL), visible=False)
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
        # viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        # viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))
        # viewer.doc.addRenderer('ext force', yr.WideArrowRenderer(rd_ext_force, rd_ext_force_pos, lineWidth=.1, fromPoint=False))
        # viewer.doc.addRenderer('trajectory', yr.LinesRenderer(rd_traj))
        # viewer.doc.addRenderer('ref_trajectory', yr.LinesRenderer(ref_traj, (0, 0, 255)))
        # viewer.doc.addRenderer('frame_label',
        #                        yr.TextRenderer(rd_frame_text_label, rd_frame_text_label_pos, text_size=30))
        # viewer.doc.addRenderer('frame_text', yr.TextRenderer(rd_frame_text, rd_frame_text_pos, text_size=30))
        # viewer.doc.addRenderer('contact_label',
        #                        yr.TextRenderer(rd_cotact_label_text, rd_cotact_label_text_pos, text_size=30))
        # viewer.doc.addRenderer('contact_info',
        #                        yr.TextRenderer(rd_contact_inf_text, rd_contact_inf_text_pos, text_size=30))
        viewer.doc.addRenderer('target strike point', yr.PointsRenderer(rd_target_zone))
        viewer.doc.addRenderer('ball_trajectory', yr.LinesRenderer(rd_ball_traj, (100,100,100), lineWidth=7.0))

        ##
        viewer.doc.addRenderer("bicep", yr.TrajRenderer(left_bicep_size, left_bicep_T, colors=render_color))
        viewer.doc.addRenderer('forearm', yr.TrajRenderer(left_forearm_size, left_forearm_T, colors=render_color))
        viewer.doc.addRenderer('hand', yr.TrajRenderer(left_hand_size, left_hand_T, colors=render_color))


    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(frame))

    def simulateCallback(frame):
        rd_frame_text[0] = str(frame)
        if ppo.env.p_fc is not None:
            # print("ppo.env.p_fc", ppo.env.p_fc)
            # print("ppo.env.p_fc hat", ppo.env.p_fc_hat)
            rd_contact_inf_text[0] = str(ppo.env.p_fc_hat)

        # print("right foot height: ", ppo.env.skel.joint(6).position_in_world_frame()[1])
        state = ppo.env.state()
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        # if frame % 60 == 59:
        #     ppo.env.ext_force = np.array([0., 0., 400.])
        #     ppo.env.ext_force_duration = 0.1
        res = ppo.env.step(action[0])
        # res = ppo.env.step(np.zeros_like(action[0]))

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make skmo file
        motion.append(q[0], dq[0])
        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        if res[2]:
            viewer.motionViewWnd.pause()
        #     print(frame, 'Done')
        #     ppo.env.reset()

        # contact rendering
        contacts = ppo.env.world.collision_result.contacts
        del rd_contact_forces[:]
        del rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                rd_contact_forces.append(-contact.f/1000.)
            else:
                rd_contact_forces.append(contact.f/1000.)
            rd_contact_positions.append(contact.p)


        # left arm trajectory
        currentSkelState = dartSkelRenderer.getState()
        
        if frame%10 ==0 or 105 > frame > 75:
            # not mound
            # left_bicep_size.append(currentSkelState[12][3])
            # left_bicep_T.append(currentSkelState[12][2])

            # left_forearm_size.append(currentSkelState[13][3])
            # left_forearm_T.append(currentSkelState[13][2])
            
            # left_hand_size.append(currentSkelState[14][3])
            # left_hand_T.append(currentSkelState[14][2])


            # on mound
            left_bicep_size.append(currentSkelState[14][3])
            left_bicep_T.append(currentSkelState[14][2])

            left_forearm_size.append(currentSkelState[15][3])
            left_forearm_T.append(currentSkelState[15][2])
            
            left_hand_size.append(currentSkelState[16][3])
            left_hand_T.append(currentSkelState[16][2])
            
            if frame > 85:
                render_color.append((255, 255, 0))
            elif frame > 75:
                render_color.append((255, 255, 255-2*frame)) 
            else:
                render_color.append((255, 255, 255- 3*frame))

        # com rendering
        del rd_COM[:]
        com = ppo.env.skel.com()
        com[1] = 0.
        rd_COM.append(com)
        # print("com:", com)

        # target point rendering
        del rd_target_zone[:]
        target_pos = ppo.env.target_zone_pos
    
        rd_target_zone.append(target_pos)

        rd_traj.append(ppo.env.skel.com())
        ref_traj.append(ppo.env.ref_skel.com())

        if ppo.env.ball_released_frame < ppo.env.current_frame:
            rd_ball_traj.append(ppo.env.ball.com())
        # rd_ball_traj.append(ppo.env.ball.com())
        # ext force rendering
        del rd_ext_force[:]
        del rd_ext_force_pos[:]
        if ppo.env.ext_force_duration > 0.:
            rd_ext_force.append(ppo.env.ext_force/500.)
            rd_ext_force_pos.append(ppo.env.skel.body('h_spine').to_world())

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-2)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(ppo.env.motion_len-2)
        CAMERA_TRACKING = False
        if CAMERA_TRACKING:
            cameraTargets = [None] * (viewer.getMaxFrame()+1)

        def postFrameCallback_Always(frame):
            if CAMERA_TRACKING:
                if cameraTargets[frame] is None:
                    cameraTargets[frame] = ppo.env.skel.body(0).com()
                viewer.setCameraTarget(cameraTargets[frame])

        viewer.setPostFrameCallback_Always(postFrameCallback_Always)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()

    # dart2bvh(bvh_file_name, ppo.env.skel, bvh_qs, 30)
    # skelqs2bvh(bvh_file_name, ppo.env.skel, bvh_qs)
    # motion.save(skmo_file_name)


if __name__ == '__main__':
    main()
