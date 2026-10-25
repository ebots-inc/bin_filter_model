// Run with: node bin_filter_offline/tests/test_layout_editor_ui.js
// Exercises scene-specific capture gating and review invalidation without hardware.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'bin_setup.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('refresh();setInterval(refresh,1000);', '');
function check(value, message) { if (!value) throw new Error(message); }
class Element {
  constructor() {
    this.value = ''; this.checked = false; this.dataset = {}; this.children = [];
    this.style = {};
    this.disabled = false; this.files = []; this.classList = {add(){}, remove(){}, toggle(){}};
  }
  replaceChildren(...items) { this.children = items; }
  querySelectorAll() { return []; }
  add(item) { this.children.push(item); }
  append(...items) { this.children.push(...items); }
  setAttribute(name, value) { this[name]=value; }
}
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
},createElement(){return new Element();},createElementNS(){return new Element();},createTextNode(value){return {textContent:value};}};
function Option(label, value) { this.label = label; this.value = value; }
const api = new Function('document', 'Option', source + `
  return {setState(value){state=value;connected=true;}, setDraft(value){layoutDraft=value;},
    render, renderButtons, layoutNamesValid, layoutRectangle, layoutPointer, layoutSelectionRequest,
    invalidateLayoutAnalysis, getDraft(){return layoutDraft;},
    request(id){let request;command=(name,values)=>{request={command:name,...values};};$(id).onclick();return request;}};
`)(document, Option);
const element=id=>document.getElementById(id);
const state={busy:false,session:{scene_id:0},current_stage:{id:'empty'},ready_token:'fresh-confirmation',
  camera_connection:{ready:true,scene_id:0},catalog:{active:{history:[]}}};
element('sceneInput').value='4';
element('cameraCalibration').value='calibration/deployment/camera/scene4.yaml';
element('robotCalibration').value='calibration/deployment/robot/A1.yaml';
element('layoutEmpty').checked=true;
api.setState(state);
api.renderButtons();
check(element('layoutCapture').disabled, 'An open scene-0 session cannot authorize a scene-4 layout capture');
check(!element('ready').disabled, 'Scene-0 readiness still authorizes its open collection stage');
check(api.request('checkConnection').scene_id===0, 'The collection camera check uses the open session, independent of new-layout sidebar choices');
check(api.request('layoutCheck').scene_id===4, 'The editor camera check uses the selected new-layout scene');
state.camera_connection.scene_id=4;
api.renderButtons();
check(!element('layoutCapture').disabled, 'Selected scene readiness authorizes a layout capture without a layout or completed collection');
check(element('ready').disabled, 'Checking scene 4 must not enable capture of the open scene-0 collection');
element('layoutEmpty').checked=false;
api.renderButtons();
check(element('layoutCapture').disabled, 'Layout capture requires explicit empty-bin preparation confirmation');
element('layoutEmpty').checked=true;
const capture=api.request('layoutCapture');
check(capture.command==='layout_preview'&&capture.scene_id===4&&capture.empty_confirmed===true&&capture.ready_token==='fresh-confirmation',
  'Capture request carries selected context and fresh empty preparation confirmation');
const draft={context:'selected-scene',preview:{path:'/data/layout_previews/preview-01'},bins:[
  {bin_id:'left',rectangle:[10,20,110,220]},{bin_id:'right',rectangle:[130,20,230,220]}],selected:0,
  analysis:{path:'/data/layout_previews/preview-01/review-01'},drag:null};
api.setDraft(draft);
check(api.layoutNamesValid(), 'Distinct operator bin names are allowed');
draft.bins[1].bin_id='left';
check(!api.layoutNamesValid(), 'Duplicate bin names cannot be analyzed');
draft.bins[1].bin_id='../right';
check(!api.layoutNamesValid(), 'Path characters cannot become bin names');
draft.bins[1].bin_id='2nd_bin';
check(!api.layoutNamesValid(), 'Names must start with a letter, matching backend validation');
draft.bins[1].bin_id='a'.repeat(65);
check(!api.layoutNamesValid(), 'Names longer than 64 characters cannot be analyzed');
draft.bins[1].bin_id='right';
const request=api.layoutSelectionRequest();
check(request.preview_path===draft.preview.path&&request.bins.length===2, 'Analysis uses the frozen preview with all selected bins');
request.bins[0].rectangle[0]=99;
check(draft.bins[0].rectangle[0]===10, 'Outgoing geometry cannot mutate the reviewed draft');
check(api.layoutRectangle([100.1,200.4],[10.2,20.6]).join(',')==='10,21,100,200', 'Reverse drawing normalizes to original preview pixel coordinates');
draft.preview.width=300;draft.preview.height=250;
element('layoutCanvas').createSVGPoint=()=>({matrixTransform(){return {x:this.x,y:this.y};}});
element('layoutCanvas').getScreenCTM=()=>({inverse(){return {};}});
check(api.layoutPointer({clientX:301,clientY:260}).join(',')==='299,249', 'Dragging beyond the image clamps to the final valid pixel, not width or height');
element('layoutName').value='Two resistor bins';
element('layoutReviewed').checked=false;
api.renderButtons();
check(element('layoutSave').disabled, 'Analysis alone does not authorize saving');
element('layoutReviewed').checked=true;
api.renderButtons();
check(!element('layoutSave').disabled, 'A named and explicitly reviewed analysis can be saved');
const save=api.request('layoutSave');
check(save.analysis_path===draft.analysis.path&&save.confirmed===true&&save.name==='Two resistor bins', 'Save identifies exactly the reviewed analysis');
api.invalidateLayoutAnalysis();
api.renderButtons();
check(api.getDraft().analysis===null&&!element('layoutReviewed').checked&&element('layoutSave').disabled,
  'Changing boxes or names invalidates the earlier review and save confirmation');
state.busy=true;
api.renderButtons();
check(element('layoutCapture').disabled&&element('layoutAnalyze').disabled&&element('layoutAdd').disabled,
  'An active command blocks new captures and layout mutations');
// Polling before the first collection must not reload the same preview and erase drawn boxes.
const firstSetup={busy:false,session:null,session_path:null,current_stage:null,stages:[],logs:[],
  camera_connection:{ready:true,scene_id:4},catalog:{sessions:[],active:{history:[]}},
  result:{layout_preview:{path:'/data/layout_previews/first',scene_id:4,frame_id:'A1',width:300,height:250,preview_png:'test'}}};
api.setState(firstSetup);
api.render();
api.getDraft().bins[0].bin_id='resistors';
api.getDraft().bins[0].rectangle=[10,20,200,210];
api.render();
check(api.getDraft().bins[0].bin_id==='resistors'&&api.getDraft().bins[0].rectangle.join(',')==='10,20,200,210',
  'Repeated state polls with no open collection preserve operator names and drawn boxes');
// Refresh after saving restores the saved scene/frame before selecting its layout.
const camera0='calibration/deployment/camera/scene0.yaml',camera4='calibration/deployment/camera/scene4.yaml';
const robot0='calibration/deployment/robot/B2.yaml',robot4='calibration/deployment/robot/A1.yaml';
firstSetup.catalog.configuration={scenes:[0,4],defaults:{scene_id:0,camera_calibration:camera0,robot_calibration:robot0,bin_config:'configs/default.yaml'},
  camera_calibrations:[{path:camera0,label:'Camera 0'},{path:camera4,label:'Camera 4'}],
  robot_calibrations:[{path:robot0,label:'B2',scene_ids:[0]},{path:robot4,label:'A1',scene_ids:[4]}],
  eyes_workflow:{scenes:[{scene_id:0,camera_calibration:camera0},{scene_id:4,camera_calibration:camera4}]},
  layouts:[{path:'configs/default.yaml',label:'Original',scene_id:0,frame_id:'B2',bin_ids:['original']},
    {path:'/data/layouts/resistors.yaml',label:'Resistor station',scene_id:4,frame_id:'A1',bin_ids:['resistors']}],cad_references:[]};
element('sceneInput').value='0';element('robotCalibration').value=robot0;element('cameraCalibration').value=camera0;
firstSetup.result={layout_saved:{path:'/data/layouts/resistors.yaml',label:'Resistor station',scene_id:4,frame_id:'A1',camera_calibration:camera4,robot_calibration:robot4}};
api.render();
check(element('sceneInput').value==='4'&&element('cameraCalibration').value===camera4&&element('robotCalibration').value===robot4,
  'Reloading a saved layout result restores its scene and calibration instead of keeping the station default');
check(element('layoutInput').value==='/data/layouts/resistors.yaml', 'Saved layout selection survives a browser refresh before creating the collection');
console.log('Layout editor UI checks passed');
