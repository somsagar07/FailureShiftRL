'use strict';

// A local SVG graph: clusters organize combinations, edges connect each to its input.
class CombinationGraph {
  constructor(svg,onSelect) {
    this.svg=svg;this.onSelect=onSelect;this.combinations=[];this.drag=null;this.suppressClick=false;
    this.control=id=>document.getElementById(id);
    this.control('graph-zoom-in').addEventListener('click',()=>this.zoom(1.3));
    this.control('graph-zoom-out').addEventListener('click',()=>this.zoom(1/1.3));
    this.control('graph-fit').addEventListener('click',()=>this.fit());
    this.control('graph-group-by').addEventListener('change',()=>{
      const dimension=this.control('graph-group-by').value;
      if(!this.data||!this.enabledDimensions.includes(dimension))return;
      this.groupBy=dimension;this.render();
    });
    this.control('graph-search').addEventListener('input',()=>this.filter());
    this.control('graph-search').addEventListener('keydown',event=>{
      if(event.key==='Enter') {
        event.preventDefault();const match=this.combinations.find(node=>node.visible);
        if(match){match.element.focus();this.activate(match);}
      }
    });
    svg.addEventListener('wheel',event=>{
      if(!this.view)return;
      event.preventDefault();this.zoom(event.deltaY<0?1.15:1/1.15,this.point(event));
    },{passive:false});
    svg.addEventListener('pointerdown',event=>{
      if(!this.view||this.drag||event.button!==0)return;
      const element=event.target.closest('[data-node]');
      const node=element?[...this.combinations,this.root].find(node=>node.id===element.dataset.node):null;
      this.suppressClick=false;
      this.drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,node,moved:false};
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove',event=>{
      if(!this.drag||this.drag.id!==event.pointerId)return;
      const drag=this.drag;
      if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<4)return;
      drag.moved=true;svg.classList.toggle('dragging',true);
      const scale=this.scale(),dx=(event.clientX-drag.x)/scale,dy=(event.clientY-drag.y)/scale;
      if(drag.node) {
        const node=drag.node;node.x+=dx;node.y+=dy;
        node.element.setAttribute('transform',`translate(${node.x} ${node.y})`);
        if(node.edge){node.edge.setAttribute('x2',node.x);node.edge.setAttribute('y2',node.y);}
        else for(const combination of this.combinations){combination.edge.setAttribute('x1',node.x);combination.edge.setAttribute('y1',node.y);}
      } else {
        this.view.x-=dx;this.view.y-=dy;this.renderView();
      }
      drag.x=event.clientX;drag.y=event.clientY;
    });
    const endDrag=event=>{
      if(!this.drag||this.drag.id!==event.pointerId)return;
      const drag=this.drag;
      this.drag=null;svg.classList.toggle('dragging',false);
      this.suppressClick=true;
      if(event.type!=='lostpointercapture'&&svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);
      // Pointer capture can retarget the browser's click to the SVG. Select here
      // and ignore that follow-up click so a drag never opens a different sample.
      if(event.type==='pointerup'&&drag.node&&!drag.moved){drag.node.element.focus();this.activate(drag.node);}
    };
    for(const name of ['pointerup','pointercancel','lostpointercapture'])svg.addEventListener(name,endDrag);
    svg.addEventListener('keydown',event=>{
      if(event.target!==svg||!this.view)return;
      if(event.key==='+'||event.key==='=')this.zoom(1.3);
      else if(event.key==='-')this.zoom(1/1.3);
      else if(event.key==='0')this.fit();
      else if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
        this.view.x+=event.key==='ArrowLeft'?-this.view.width*.08:event.key==='ArrowRight'?this.view.width*.08:0;
        this.view.y+=event.key==='ArrowUp'?-this.view.height*.08:event.key==='ArrowDown'?this.view.height*.08:0;
        this.renderView();
      } else return;
      event.preventDefault();
    });
  }

  svgElement(tag,attributes,text) {
    const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
    if(text!==undefined)node.textContent=text;
    return node;
  }

  setData(data,task,describe,actionSpace={}) {
    this.data=data;this.task=task;this.describe=describe;this.selectedId='original';
    const dimensions={rotation:'Rotation',darkness:'Darkness',saturation:'Saturation'};
    this.enabledDimensions=Object.keys(dimensions).filter(key=>actionSpace[key]?.length!==0);
    const selector=this.control('graph-group-by');selector.replaceChildren();
    for(const [key,label] of Object.entries(dimensions)) {
      const option=document.createElement('option');option.value=key;option.disabled=!this.enabledDimensions.includes(key);
      option.textContent=label+(option.disabled?' (off)':'');selector.append(option);
    }
    if(!this.enabledDimensions.includes(this.groupBy))this.groupBy=this.enabledDimensions.find(key=>new Set(data.map(node=>node.action[key])).size>1)||this.enabledDimensions[0];
    selector.value=this.groupBy||'';selector.disabled=!this.enabledDimensions.length;
    this.control('graph-grouping').hidden=task!=='classification';
    this.control('graph-space').hidden=task!=='classification';
    this.control('graph-space').textContent=Object.entries(dimensions).map(([key,label])=>`${label}: ${this.enabledDimensions.includes(key)?new Set(data.map(node=>node.action[key])).size:'Off'}`).join(' · ');
    this.control('graph-search').value='';this.render();
  }

  render() {
    if(this.drag) {
      const id=this.drag.id;this.drag=null;
      if(this.svg.hasPointerCapture(id))this.svg.releasePointerCapture(id);
      this.svg.classList.toggle('dragging',false);
    }
    const palette=['#336aa3','#9a603a','#598341','#8865a5','#24847d'];
    const dimensions={rotation:'Rotation',darkness:'Darkness',saturation:'Saturation'};
    const dimension=this.groupBy;
    const groups=new Map();
    for(const node of this.data) {
      const label=this.task==='classification'?(dimension?`${dimensions[dimension]} ${node.action[dimension]}${dimension==='rotation'?'°':''}`:'Combinations'):this.task==='generation'?this.describe(node.action):'Text edits';
      if(!groups.has(label))groups.set(label,[]);
      groups.get(label).push(node);
    }
    this.svg.replaceChildren();this.combinations=[];this.groups=[];
    this.control('graph-legend').replaceChildren();
    const edges=this.svgElement('g',{'aria-hidden':'true'}),labels=this.svgElement('g',{'aria-hidden':'true'}),nodes=this.svgElement('g',{});
    this.svg.append(edges,labels,nodes);
    const maxSize=Math.max(1,...[...groups.values()].map(group=>group.length));
    const orbit=groups.size>1?Math.max(150,(19*Math.sqrt(maxSize)+24)/Math.sin(Math.PI/groups.size)+25):0;
    [...groups.entries()].forEach(([label,items],groupIndex)=>{
      const angle=-Math.PI/2+groupIndex*2*Math.PI/groups.size;
      const center={x:Math.cos(angle)*orbit,y:Math.sin(angle)*orbit},color=palette[groupIndex%palette.length];
      const extent=groups.size===1?80+14*Math.sqrt(items.length):19*Math.sqrt(items.length);
      const groupLabel=this.svgElement('text',{x:center.x,y:center.y-extent-25,class:'group-label'},label);
      labels.append(groupLabel);
      const group={element:groupLabel,nodes:[],x:center.x,y:center.y-extent-25};this.groups.push(group);
      items.forEach((data,index)=>{
        const angle=index*Math.PI*(3-Math.sqrt(5));
        const radius=groups.size===1?80+14*Math.sqrt(index):19*Math.sqrt(index+.5);
        const node={data,id:String(data.id),x:center.x+Math.cos(angle)*radius,y:center.y+Math.sin(angle)*radius,color,visible:true};
        node.description=`${this.task==='generation'?`Action ${data.id}`:`Combination ${data.id+1}`}: ${this.describe(data.action)}`;
        node.edge=this.svgElement('line',{x1:0,y1:0,x2:node.x,y2:node.y,stroke:color,class:'edge'});
        edges.append(node.edge);node.element=this.makeNode(node);nodes.append(node.element);
        this.combinations.push(node);group.nodes.push(node);
      });
      const item=document.createElement('span'),name=document.createElement('span');
      const marker=this.svgElement('svg',{width:10,height:10,'aria-hidden':'true'});
      marker.append(this.svgElement('circle',{cx:5,cy:5,r:4,fill:color}));
      name.textContent=label;item.append(marker,name);this.control('graph-legend').append(item);
    });
    this.root={id:'original',data:null,x:0,y:0,color:'#353b43',description:'Original input'};
    this.root.element=this.makeNode(this.root);nodes.append(this.root.element);
    this.fit();this.filter();this.select(this.selectedId);
  }

  makeNode(node) {
    const root=node.id==='original';
    const group=this.svgElement('g',{class:`graph-node${root?' original-node':''}`,role:'button',tabindex:0,'aria-label':node.description,'aria-pressed':'false','data-node':node.id,transform:`translate(${node.x} ${node.y})`});
    group.append(this.svgElement('title',{},node.description),this.svgElement('circle',{r:root?18:10,fill:node.color}));
    group.append(this.svgElement('text',{x:0,y:root?36:3.5,class:root?'root-label':'node-label'},root?'Original':node.data.id+(this.task==='generation'?0:1)));
    group.addEventListener('click',event=>{if(!event?.detail||!this.suppressClick)this.activate(node);});
    group.addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' '){event.preventDefault();this.activate(node);}
    });
    return group;
  }

  activate(node) {this.select(node.id);this.onSelect(node.data);}

  select(id) {
    this.selectedId=String(id);
    for(const node of [...this.combinations,this.root]) {
      const selected=node.id===this.selectedId;
      node.element.classList.toggle('selected',selected);node.element.setAttribute('aria-pressed',String(selected));
      if(node.edge)node.edge.classList.toggle('selected',selected);
    }
  }

  filter() {
    const query=this.control('graph-search').value.trim().toLowerCase();
    const number=/^#?\d+$/.test(query)?Number(query.replace('#','')):null;
    for(const node of this.combinations) {
      node.visible=!query||(number!==null?node.data.id+(this.task==='generation'?0:1)===number:node.description.toLowerCase().includes(query));
      for(const element of [node.element,node.edge])element.setAttribute('display',node.visible?'inline':'none');
    }
    for(const group of this.groups||[])group.element.setAttribute('display',group.nodes.some(node=>node.visible)?'inline':'none');
    const visible=this.combinations.filter(node=>node.visible).length;
    this.control('graph-count').textContent=query?`${visible} of ${this.combinations.length} combinations match`:`${this.combinations.length} combinations · click a node to inspect its sample`;
  }

  scale() {
    const box=this.svg.getBoundingClientRect();
    return Math.min(box.width/this.view.width,box.height/this.view.height)||1;
  }

  point(event) {
    const box=this.svg.getBoundingClientRect(),scale=this.scale();
    return {x:this.view.x+(event.clientX-box.left-(box.width-this.view.width*scale)/2)/scale,
      y:this.view.y+(event.clientY-box.top-(box.height-this.view.height*scale)/2)/scale};
  }

  zoom(factor,anchor) {
    if(!this.view)return;
    const width=Math.max(this.bounds.width/8,Math.min(this.bounds.width*2,this.view.width/factor));
    const ratio=width/this.view.width;
    anchor=anchor||{x:this.view.x+this.view.width/2,y:this.view.y+this.view.height/2};
    this.view={x:anchor.x-(anchor.x-this.view.x)*ratio,y:anchor.y-(anchor.y-this.view.y)*ratio,width,height:this.view.height*ratio};
    this.renderView();
  }

  fit() {
    if(!this.root)return;
    const points=[this.root,...this.combinations,...this.groups];
    const left=Math.min(...points.map(point=>point.x))-75,top=Math.min(...points.map(point=>point.y))-40;
    this.bounds={x:left,y:top,width:Math.max(...points.map(point=>point.x))-left+75,height:Math.max(...points.map(point=>point.y))-top+60};
    this.view={...this.bounds};this.renderView();
  }

  renderView() {
    const {x,y,width,height}=this.view;this.svg.setAttribute('viewBox',`${x} ${y} ${width} ${height}`);
    this.control('graph-zoom').textContent=`${Math.round(this.bounds.width/width*100)}%`;
  }
}
