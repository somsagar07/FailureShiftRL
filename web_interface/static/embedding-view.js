'use strict';

// A perspective camera over measured PCA coordinates. Only the camera moves;
// input objects and their embedding positions are never changed.
class EmbeddingView {
  constructor(canvas, {onSelectSample, onSelectAction, onClear} = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.onSelectSample = onSelectSample;
    this.onSelectAction = onSelectAction;
    this.onClear = onClear;
    this.points = [];
    this.nodes = [];
    this.focusIndex = null;
    this.selectedAction = null;
    this.projected = [];
    this.center = [0, 0, 0];
    this.scale = 1;
    this.yaw = 0;
    this.pitch = 0;
    this.pan = {x: 0, y: 0};
    this.distance = this.fitDistance = 4;
    this.drag = null;
    canvas.tabIndex = 0;
    canvas.style.touchAction = 'none';
    canvas.style.cursor = 'grab';
    canvas.addEventListener('pointerdown', event => this.startPointer(event));
    canvas.addEventListener('pointermove', event => this.movePointer(event));
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(type, event => this.endPointer(event));
    }
    // These also end a gesture if a browser declines pointer capture.
    window.addEventListener('pointerup', event => this.endPointer(event));
    window.addEventListener('pointercancel', event => this.endPointer(event));
    canvas.addEventListener('pointerleave', () => {
      canvas.title = '';
      if (!this.drag) canvas.style.cursor = 'grab';
    });
    canvas.addEventListener('contextmenu', event => event.preventDefault());
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.height : 1);
      this.zoom(Math.exp(Math.max(-0.5, Math.min(0.5, -pixels * 0.002))));
    }, {passive: false});
    canvas.addEventListener('keydown', event => this.keydown(event));
    window.addEventListener('resize', () => this.render());
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.render());
      this.resizeObserver.observe(canvas);
    }
    this.render();
  }

  setData(points) {
    if (this.drag) this.endPointer({type: 'pointercancel', pointerId: this.drag.id});
    this.canvas.title = '';
    this.discovery=null;this.measured=new Map();this.severities=new Map();
    this.actionStatistics=new Map();this.actionRadii=new Map();
    this.points = points;
    this.nodes = [];
    this.focusIndex = null;
    this.selectedAction = null;
    this.yaw = this.pitch = 0;
    this.fit();
  }

  focus(index) {
    if (!this.points.some(point => point.index === index)) return;
    if (index !== this.focusIndex) this.nodes = [];
    this.focusIndex = index;
    this.selectedAction = null;
    this.render();
  }

  setExpansion(index, nodes) {
    if (index !== this.focusIndex) return;
    this.nodes = nodes;
    if (!nodes.some(node => node.id === this.selectedAction)) this.selectedAction = null;
    this.render();
  }

  selectAction(id) {
    if (id !== null && !this.nodes.some(node => node.id === id)) return;
    this.selectedAction = id;
    this.render();
  }

  clearFocus() {
    this.focusIndex = null;
    this.selectedAction = null;
    this.nodes = [];
    this.render();
  }

  setDiscovery(job) {
    this.discovery=job;
    this.measured=new Map();this.severities=new Map();
    const statistics=job?.policy_statistics;
    this.actionStatistics=new Map((statistics?.actions||[]).map(action=>[action.action_id,action]));
    this.actionRadii=new Map();
    if(statistics?.policy_updates>0&&statistics.input_count>1) {
      const ranked=[...this.actionStatistics.values()].filter(action=>Number.isFinite(action.std)&&action.std>=0).sort((a,b)=>a.std-b.std);
      for(let first=0;first<ranked.length;) {
        let last=first;while(last+1<ranked.length&&ranked[last+1].std===ranked[first].std)last++;
        // Average tied ranks; all-equal values get a neutral midpoint.
        const rank=ranked.length>1?(first+last)/(2*(ranked.length-1)):.5;
        for(let i=first;i<=last;i++)this.actionRadii.set(ranked[i].action_id,11-7*rank);
        first=last+1;
      }
    }
    const groups=new Map();
    for(const record of job?.records||[]) {
      if(!Number.isFinite(record.severity))continue;
      this.measured.set(`${record.input_index}:${record.action_id}`,record.severity);
      const values=groups.get(record.input_index)||[];values.push(record.severity);groups.set(record.input_index,values);
    }
    for(const [index,values] of groups)this.severities.set(index,values.reduce((a,b)=>a+b,0)/values.length);
    this.render();
    clearTimeout(this.pulseTimer);
    if(job&&['queued','running'].includes(job.status)&&typeof setTimeout==='function'&&!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)this.pulseTimer=setTimeout(()=>this.setDiscovery(this.discovery),160);
  }

  severityColor(value) {
    if(!Number.isFinite(value))return '#9aa1a9';
    const severity=Math.max(0,Math.min(1,value));
    return `rgb(${Math.round(47+166*severity)},${Math.round(158-107*severity)},${Math.round(88-36*severity)})`;
  }

  policyHint(id) {
    if(!this.discovery)return '';
    const statistics=this.discovery.policy_statistics,action=this.actionStatistics.get(id),updated=statistics?.policy_updates>0;
    const count=statistics?.input_count??'unknown';
    const mean=updated&&Number.isFinite(action?.mean)?action.mean.toPrecision(3):'unavailable';
    const reason=!statistics?'statistics unavailable':!updated?'no policy updates yet':statistics.input_count<2?'requires at least 2 original inputs':'not available';
    const variation=this.actionRadii.has(id)?action.std.toPrecision(3):`unavailable (${reason})`;
    return ` · mean DQN action preference: ${mean} · variation across ${count} original inputs (sample SD): ${variation} · tested inputs: ${action?.evaluated_inputs??'—'}/${count} · not calibrated failure confidence`;
  }

  reset() {clearTimeout(this.pulseTimer);this.setData([]);}

  zoom(factor) {
    if (!Number.isFinite(factor) || factor <= 0) return;
    this.distance = Math.max(0.2, Math.min(this.fitDistance * 40, this.distance / factor));
    this.render();
  }

  orbit(dx, dy) {
    this.yaw += dx;
    this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch + dy));
    this.render();
  }

  pick(event) {
    const box = this.canvas.getBoundingClientRect();
    const x = event.clientX - box.left, y = event.clientY - box.top;
    if (x < 0 || y < 0 || x > this.width || y > this.height) return null;
    // Prefer a visible disk before considering the small pointer tolerance.
    for (const padding of [0, 3]) {
      for (let index = this.projected.length - 1; index >= 0; index--) {
        const point = this.projected[index];
        if (Math.hypot(x - point.x, y - point.y) <= point.radius + padding) return point;
      }
    }
    return null;
  }

  startPointer(event) {
    if (this.drag || event.isPrimary === false || ![0, 2].includes(event.button)) return;
    event.preventDefault();
    this.canvas.focus({preventScroll: true});
    this.drag = {id: event.pointerId, x: event.clientX, y: event.clientY,
      startX: event.clientX, startY: event.clientY, moved: false,
      pan: event.shiftKey || event.button === 2, point: this.pick(event), focusIndex: this.focusIndex};
    this.canvas.title = '';
    try {this.canvas.setPointerCapture(event.pointerId);} catch (_) { /* Pointer may already have ended. */ }
  }

  movePointer(event) {
    const drag = this.drag;
    if (!drag) {
      const point = this.pick(event);
      this.canvas.style.cursor = point ? 'pointer' : 'grab';
      this.canvas.title = !point ? '' : point.kind === 'sample' ? point.data.name :
        `${Object.entries(point.data.action).map(([key, value]) => `${key}: ${value}`).join(', ')} · cosine similarity: ${point.data.similarity.toFixed(3)}${this.policyHint(point.data.id)}`;
      return;
    }
    if (drag.id !== event.pointerId) return;
    if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 4) return;
    drag.moved = true;
    this.canvas.style.cursor = 'grabbing';
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (drag.pan) {
      this.pan.x += dx;
      this.pan.y += dy;
      this.render();
    } else this.orbit(dx * 0.008, dy * 0.008);
  }

  endPointer(event) {
    const drag = this.drag;
    if (!drag || drag.id !== event.pointerId) return;
    this.drag = null;
    this.canvas.style.cursor = 'grab';
    try {
      if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    } catch (_) { /* Capture can disappear when the element is hidden. */ }
    if (event.type !== 'pointerup' || drag.moved || drag.pan ||
        Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) >= 4) return;
    const point = this.pick(event);
    if (!point || point.kind !== drag.point?.kind) return;
    // Polling replaces JSON objects, but the same input/action is still a click.
    if (point.kind === 'sample' ? point.data.index !== drag.point.data.index :
        this.focusIndex !== drag.focusIndex || point.data.id !== drag.point.data.id) return;
    if (point.kind === 'sample') {
      this.focus(point.data.index);
      this.onSelectSample?.(point.data);
    } else {
      this.selectAction(point.data.id);
      this.onSelectAction?.(point.data);
    }
  }

  keydown(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '+' || event.key === '=') this.zoom(1.25);
    else if (event.key === '-') this.zoom(1 / 1.25);
    else if (event.key === '0') this.fit();
    else if (event.key === 'ArrowLeft') this.orbit(-0.12, 0);
    else if (event.key === 'ArrowRight') this.orbit(0.12, 0);
    else if (event.key === 'ArrowUp') this.orbit(0, -0.12);
    else if (event.key === 'ArrowDown') this.orbit(0, 0.12);
    else if (event.key === 'Escape') {
      this.clearFocus();
      this.onClear?.();
    } else return;
    event.preventDefault();
  }

  resize() {
    const box = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, box.width);
    this.height = Math.max(1, box.height);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.round(this.width * ratio), height = Math.round(this.height * ratio);
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    // All projection and picking coordinates are CSS pixels, independent of DPR.
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  fit() {
    this.resize();
    const points = [...this.points, ...this.nodes];
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    for (const point of points) {
      for (let axis = 0; axis < 3; axis++) {
        low[axis] = Math.min(low[axis], point.position[axis]);
        high[axis] = Math.max(high[axis], point.position[axis]);
      }
    }
    this.center = points.length ? low.map((value, axis) => value / 2 + high[axis] / 2) : [0, 0, 0];
    this.scale = 0;
    for (const point of points) this.scale = Math.max(this.scale, Math.hypot(...point.position.map((value, axis) => value - this.center[axis])));
    if (!this.scale) this.scale = 1;
    // The bounding sphere fits at any camera angle, including in a narrow panel.
    this.distance = this.fitDistance = 1 + 1 / (0.78 * Math.tan(Math.PI / 8) * Math.min(1, this.width / this.height));
    this.pan = {x: 0, y: 0};
    this.render();
  }

  rotate(position) {
    const [x, y, z] = position;
    const turnedX = Math.cos(this.yaw) * x + Math.sin(this.yaw) * z;
    const turnedZ = -Math.sin(this.yaw) * x + Math.cos(this.yaw) * z;
    return [turnedX, Math.cos(this.pitch) * y - Math.sin(this.pitch) * turnedZ,
      Math.sin(this.pitch) * y + Math.cos(this.pitch) * turnedZ];
  }

  project(data, kind) {
    const unit = data.position.map((value, axis) => (value - this.center[axis]) / this.scale);
    const [x, y, z] = this.rotate(unit);
    const depth = this.distance - z;
    if (depth <= 0.05) return null;
    const focal = this.height / (2 * Math.tan(Math.PI / 8));
    const focused = kind === 'sample' && data.index === this.focusIndex;
    const selected = kind === 'action' ? data.id === this.selectedAction : focused && this.selectedAction === null;
    const baseRadius=kind==='action'&&this.discovery?(this.actionRadii.get(data.id)??3.5):(selected ? 8 : focused ? 7 : kind === 'action' ? 6 : 5);
    const radius = Math.max(2, Math.min(18, baseRadius * this.fitDistance / depth));
    return {data, kind, depth, selected, focused, radius,
      x: this.width / 2 + this.pan.x + x * focal / depth,
      y: this.height / 2 + this.pan.y - y * focal / depth};
  }

  render() {
    this.resize();
    const ctx = this.context;
    ctx.clearRect(0, 0, this.width, this.height);
    this.projected = [
      ...this.points.map(point => this.project(point, 'sample')),
      ...this.nodes.map(node => this.project(node, 'action')),
    ].filter(Boolean).sort((a, b) => b.depth - a.depth);

    const original = this.projected.find(point => point.kind === 'sample' && point.focused);
    if (original) {
      for (const node of this.projected) {
        if (node.kind !== 'action') continue;
        ctx.beginPath();
        ctx.moveTo(original.x, original.y);
        ctx.lineTo(node.x, node.y);
        ctx.strokeStyle = '#a57143';
        ctx.lineWidth = node.selected ? 2 : 1;
        ctx.globalAlpha = node.selected ? 0.8 : 0.35;
        ctx.stroke();
      }
    }
    for (const node of this.projected) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.discovery?this.severityColor(node.kind==='sample'?this.severities.get(node.data.index):this.measured.get(`${this.focusIndex}:${node.data.id}`)):(node.kind === 'sample' ? '#336aa3' : '#b06c36');
      ctx.globalAlpha = this.focusIndex !== null && node.kind === 'sample' && !node.focused ? 0.22 : 1;
      ctx.fill();
      // A light rim separates overlapping points; selection gets a darker border.
      ctx.strokeStyle = node.selected ? '#202a35' : '#ffffff';
      ctx.lineWidth = node.selected ? 2.5 : 1.5;
      ctx.setLineDash(this.discovery&&node.kind==='action'&&!this.actionRadii.has(node.data.id)?[2,2]:[]);
      ctx.stroke();
      ctx.setLineDash([]);
      const current=this.discovery?.current;
      const active=this.discovery&&['queued','running'].includes(this.discovery.status)&&current&&
        (node.kind==='sample'?node.data.index===current.input_index:this.focusIndex===current.input_index&&node.data.id===current.action_id);
      if(active) {
        const pulse=2+2*(1+Math.sin(Date.now()/220));
        ctx.beginPath();ctx.arc(node.x,node.y,node.radius+pulse,0,Math.PI*2);
        ctx.strokeStyle='#203f72';ctx.lineWidth=2;ctx.globalAlpha=0.85;ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
}
