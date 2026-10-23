"""Small synthetic camera/session fixture for Python and native replay tests.

These rendered bumps exercise geometry and stage plumbing. They do not claim to
model a production resistor or demonstrate real factory acceptance.
"""
from pathlib import Path
import importlib.util
import json
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import dataset as data
from test_modern_model import settings, surface


def create_session(root):
    import OpenEXR
    root = Path(root)
    artifacts = root / 'artifacts'
    rows, cols, focal, cx, cy = 240, 560, 400., 128., 120.
    intrinsic = [focal,0,cx,0,focal,cy,0,0,1]
    cameras = {name:{'rows':rows,'cols':cols,'cam_intrinsic':intrinsic,'cam_distortion':[0.]*5,
                     'cam_extrinsic':np.eye(4).ravel().tolist()} for name in ('cam1','cam2')}
    cameras.update(world_r=np.eye(3).ravel().tolist(),world_t=[0,0,0])
    camera_path = 'calibration/deployment/camera/scene0_denali0_params.yaml'
    robot_path = 'calibration/deployment/robot/B2_params.yaml'
    data.write_yaml(artifacts / camera_path, cameras)
    data.write_yaml(artifacts / robot_path, {'name':'B2','tools':[], 'hand_eye':[{'scene_id':0,'calibration':[0,0,-100,0,0,0]}]})
    data.write_yaml(artifacts / 'bin_config.yaml', settings())
    data.write_yaml(artifacts / 'workflow_configs/bin_calibration.yaml', {'scenes': [{'scene_id': 0,
        'camera': [{'cam_id': 0, 'parameters': ['artifacts/' + camera_path]}]}]})
    recipe = {'scene_id':0,'denali_id':0,'frame_id':'B2','length_unit':'mm','exposure_ms':[4.,20.],
              'empty_train_rounds':3,'empty_test_rounds':1,'loaded_test_rounds':1,
              'gain':[1.,1.],'black_level':[0,0],'camera_calibration':camera_path,
              'robot_calibration':robot_path,'bin_config':'bin_config.yaml',
              'guided':{'loaded_arrangements':1,
                        'normal_fill_instruction':'Several synthetic bumps.', 'full_fill_instruction':'Several larger synthetic bumps.',
                        'acceptance':{'minimum_part_thickness_mm':None}}}
    helper = HERE.parent / 'ebots_system_artifacts/artifacts_cell_4/data_saving/bin_model/prepare_session.py'
    spec = importlib.util.spec_from_file_location('modern_collection_fixture', helper)
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    folder = frontend.prepare_session(artifacts,root/'sessions',root/'sessions',recipe,mode='guided',
                                      name='synthetic-bin-setup')
    session = json.loads((folder/'session.json').read_text())
    session['synthetic'] = True
    data.write_json(folder/'session.json',session)
    stages = {stage['id']:stage for stage in session['stages']}
    (folder/'stage_confirmations').mkdir()
    for stage in stages.values():
        data.write_json(folder/stage['confirmation_path'], {'session_id':session['session_id'],
            'stage_id':stage['id'],'gate_token':'synthetic-test-only','confirmed_at':'2026-10-23T10:00:00Z'})
    yy,xx = np.indices((rows,cols)); rx=(xx-cx)/focal; ry=(yy-cy)/focal
    for index,capture in enumerate(session['captures']):
        directory = folder/capture['path']
        directory.mkdir(parents=True)
        (directory/'calibration').mkdir()
        for key in ('camera','robot'):
            shutil.copyfile(artifacts/session['calibration'][key]['artifact_path'],directory/f'calibration/{key}.yaml')
        stage=stages[capture['stage_id']]
        manifest={'status':'complete','simulated':False,'scene_id':0,'frame_id':'B2','length_unit':'mm','depth_axis':'z',
                  'part_name':capture['part_name'],'task_name':'capture','images':{},'files':[],
                  'calibration':{key:{'path':f'calibration/{key}.yaml'} for key in ('camera','robot')},'requested_settings':{'path':'requested_settings.yaml',
                      'values_are_requested_not_sensor_verified':True}}
        for camera in (0,1):
            depth=np.full((rows,cols),100.)
            for _ in range(40):
                x,y=rx*depth,ry*depth
                local_x=np.where(x>40,x-80,x)
                updated=100+surface(local_x,y)
                for bin_id,offset in [('bin_a',0.),('bin_b',80.)]:
                    if capture['bins'][bin_id]!='loaded':
                        continue
                    if stage['bins'][bin_id].get('placement') == 'wall':
                        side = stage['bins'][bin_id].get('wall_side', 'left')
                        px, py = {'left': (-8.5, 0.), 'right': (8.5, 0.),
                                  'bottom': (0., -8.5), 'top': (0., 8.5)}[side]
                        peaks=[(px,py,6.)]
                    elif 'corner' in stage['id']:
                        peaks=[(-9.,-8.,6.)]
                    elif 'normal' in stage['id']:
                        peaks=[(-5.,0.,6.),(5.,3.,7.)]
                    elif 'full' in stage['id']:
                        peaks=[(-5.,-3.,8.),(5.,4.,9.)]
                    else:
                        peaks=[(0.,0.,6.)]
                    for px,py,height in peaks:
                        d2=(x-offset-px)**2+(y-py)**2
                        updated += height*np.exp(-d2/(2*2.5**2))*(d2<25)
                previous,depth=depth,updated
            stable=np.abs(previous-depth)<.05
            bias=(-.5+.03*capture['exposure_ms']) if camera==0 else (.4+.06*capture['exposure_ms'])
            measured=(depth+bias+np.random.default_rng(100+index*2+camera).normal(0,.025,depth.shape)).astype(np.float32)
            measured[~stable]=0
            for kind in ('depth','spotlight'):
                path=directory/kind/f'Frame{camera}.exr'; path.parent.mkdir(parents=True,exist_ok=True)
                OpenEXR.File({}, {'Y': measured if kind=='depth' else np.ones(measured.shape,dtype=np.float32)}).write(str(path))
                manifest['images'].setdefault(kind,[]).append({'height':rows,'width':cols,'dtype':'float32'})
        data.write_yaml(directory/'requested_settings.yaml', {'structured_light_exposure_ms':[capture['exposure_ms']]*2,
            'spotlight_exposure_ms':[capture['spotlight_exposure_ms']]*2,'gain':[1.,1.],'black_level':[0,0],
            'values_are_requested_not_sensor_verified':True})
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                manifest['files'].append({'path':str(path.relative_to(directory)),'size_bytes':path.stat().st_size})
        data.write_yaml(directory/'capture_manifest.yaml',manifest)
    frontend.verify_session(folder)
    return folder


if __name__=='__main__':
    print(create_session(Path(sys.argv[1])))
