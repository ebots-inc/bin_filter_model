// Run with: node bin_filter_offline/tests/test_cad_reference_ui.js
// Uses a small DOM substitute to test session filtering without a browser.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'bin_setup.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('refresh();setInterval(refresh,1000);', '');
function check(value, message) { if (!value) throw new Error(message); }
class Element {
  constructor() { this.value = ''; this.dataset = {}; this.children = []; this.disabled = false; }
  replaceChildren(...items) {
    this.children = items;
    this.value = items[0]?.value || '';
  }
  querySelectorAll() { return []; }
  add(item) { this.children.push(item); }
}
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
}};
function Option(label, value) { this.label = label; this.value = value; }
const api = new Function('document', 'Option', source + `
  return {setState(value){state=value;connected=true;}, renderCadReferences, renderButtons,
    buildRequest(){let request;command=(name,values)=>{request={name,...values};};$('build').onclick();return request;}};
`)(document, Option);
const item = (path, scene, frame, bins) => ({path, label:path, scene_id:scene, frame_id:frame, bin_ids:bins});
const references = [item('matching.yaml', 0, 'B2', ['bin_a', 'bin_b']),
  item('other-scene.yaml', 4, 'B2', ['bin_a', 'bin_b']),
  item('other-frame.yaml', 0, 'A1', ['bin_a', 'bin_b']),
  item('missing-bin.yaml', 0, 'B2', ['bin_a'])];
const state = {session_path:'/sessions/current', session:{scene_id:0, frame_id:'B2', bin_ids:['bin_a', 'bin_b']},
  catalog:{configuration:{cad_references:references}}, current_stage:null, busy:false};
// The pending New setup fields intentionally describe a different scene.
document.getElementById('sceneInput').value = '4';
document.getElementById('robotCalibration').value = 'A1';
api.setState(state);
api.renderCadReferences();
const select = document.getElementById('cadReference');
check(select.children.map(item=>item.value).join(',') === ',matching.yaml', 'Filter by open session, including every bin');
check(select.value === '', 'A new session defaults to measured captures');
select.value = 'matching.yaml';
api.renderCadReferences();
check(select.value === 'matching.yaml', 'Polling must preserve an explicit selection');
check(api.buildRequest().cad_reference === 'matching.yaml', 'Build must forward the selected plan');
state.cad={references:[item('session:cad_previews/preview-01/plan.yaml',0,'B2',['bin_a','bin_b'])]};
api.renderCadReferences();
check(select.children.some(item=>item.value==='session:cad_previews/preview-01/plan.yaml'), 'Confirmed uploaded references join available choices');
select.value='session:cad_previews/preview-01/plan.yaml';
check(api.buildRequest().cad_reference==='session:cad_previews/preview-01/plan.yaml', 'Build forwards the session-local CAD reference');
api.renderButtons();
check(!select.disabled, 'An idle guided session can choose CAD');
state.busy = true;
api.renderButtons();
check(select.disabled, 'CAD selection is disabled while busy');
state.busy = false;
state.session.mode = 'check';
api.renderButtons();
check(select.disabled, 'Quick checks do not fit CAD');
state.session.mode = 'guided';
state.session_path = '/sessions/another';
api.renderCadReferences();
check(select.value === '', 'Opening another session clears the previous CAD choice');
check(api.buildRequest().cad_reference === '', 'Measured-only builds send an empty selection');
state.session_path = null;
state.session = null;
api.renderCadReferences();
api.renderButtons();
check(select.disabled && select.children.length === 1, 'No session exposes no CAD plan');
console.log('CAD reference UI checks passed');
