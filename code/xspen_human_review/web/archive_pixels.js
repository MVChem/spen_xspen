'use strict';
// Local script chunks keep file:// browsing possible without loading entire volumes.
window.ArchivePixels=(()=>{
  const cache=new Map(),pending=new Map();
  function key(s,n){return `${s.chunk_key}:${n}`;}
  function decode(text){const binary=atob(text),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);return bytes;}
  function load(s,n){const id=key(s,n);if(cache.has(id))return Promise.resolve(cache.get(id));if(pending.has(id))return pending.get(id);
    const promise=new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=s.chunks[n];
      script.onload=()=>{try{const item=window.REVIEW_CHUNKS[id],bytes=decode(item.data);if(new Uint8Array(new Uint16Array([1]).buffer)[0]===1)item.pixels=new Uint16Array(bytes.buffer);else{const view=new DataView(bytes.buffer);item.pixels=new Uint16Array(bytes.length/2);for(let i=0;i<item.pixels.length;i++)item.pixels[i]=view.getUint16(i*2,true);}delete item.data;if(item.invalid)item.invalidPixels=decode(item.invalid);delete item.invalid;cache.set(id,item);delete window.REVIEW_CHUNKS[id];pending.delete(id);script.remove();resolve(item);}catch(error){pending.delete(id);script.remove();reject(error);}};
      script.onerror=()=>{pending.delete(id);script.remove();reject(new Error('无法读取部分图像数据，请确认输出文件夹完整。'));};document.head.append(script);
    });pending.set(id,promise);return promise;
  }
  async function ensure(requests){const unique=new Map();for(const [s,index] of requests)if(s.chunked){const n=Math.floor(index/s.chunk_size);unique.set(key(s,n),[s,n]);}
    const queue=[...unique.values()];let next=0;await Promise.all(Array.from({length:Math.min(6,queue.length)},async()=>{while(next<queue.length){const [s,n]=queue[next++];await load(s,n);}}));return new Set(unique.keys());}
  function read(s,index){if(!s.chunked)return {pixels:s.pixels,offset:index*s.shape[1]*s.shape[2],min:0,max:s.maxima[index],p995:s.p995[index],p005:0};
    const n=Math.floor(index/s.chunk_size),item=cache.get(key(s,n));if(!item)throw new Error('图像分块尚未加载');const local=index-item.start;
    return {pixels:item.pixels,offset:local*s.shape[1]*s.shape[2],min:item.minima[local],max:item.maxima[local],p995:item.p995[local],p005:item.p005[local],invalid:item.invalidPixels};
  }
  function prune(keep){if(cache.size<=96)return;for(const k of cache.keys()){if(cache.size<=96)break;if(!keep.has(k))cache.delete(k);}}
  return {ensure,read,prune,cacheSize:()=>cache.size};
})();
