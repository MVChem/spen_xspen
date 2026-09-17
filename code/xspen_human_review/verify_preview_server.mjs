// Isolated harness using the installed chat app's real preview-serving code.
// No production cookies, database, configuration or server are changed.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { resolve, join } from 'node:path';
const [chatRootArg, folderArg] = process.argv.slice(2);
if (!chatRootArg || !folderArg) throw new Error('Usage: node verify_preview_server.mjs CHAT_ROOT EXPERIMENT_FOLDER');
const chatRoot = resolve(chatRootArg), folder = resolve(folderArg);
const require = createRequire(join(chatRoot, 'package.json'));
const express = require('express');
const { HtmlPreviews } = await import(pathToFileURL(join(chatRoot, 'dist/server/htmlPreviews.js')));
const { Artifacts } = await import(pathToFileURL(join(chatRoot, 'dist/server/artifacts.js')));
const session = { id:'local-preview-verification', user_id:'local-verifier', workspace:'review', cwd:folder, source:'web' };
const config = {databasePath:join(folder,'unused-test-db'),workspaces:[{id:'review',path:folder}],codexSessionRoot:folder};
const store = {user:{id:session.user_id},session:(id,user)=>id===session.id&&user===session.user_id?session:undefined};
const previews = new HtmlPreviews(store, new Artifacts(config), hash=>hash==='isolated-verification');
const previewUrl = previews.url(session.id, 'isolated-verification', join(folder,'index.html'));
const app = express();
app.use((_req,res,next)=>{res.set('Cache-Control','no-store');next();});
app.get('/api/file-previews/:token/*path', previews.serve);
app.get('/preview-info',(_req,res)=>res.json({previewUrl}));
app.get('/',(_req,res)=>res.type('html').send(`<!doctype html><html><head><meta charset="utf-8"><title>Chat preview verification</title><style>body{margin:0}iframe{width:100%;height:100vh;border:0}</style></head><body><iframe title="HTML preview" sandbox="allow-scripts" referrerpolicy="no-referrer" src="${previewUrl}"></iframe></body></html>`));
app.use((error,_req,res,_next)=>res.status(error.status||500).json({error:error.message}));
const server=app.listen(0,'127.0.0.1',()=>process.stdout.write(JSON.stringify({baseUrl:`http://127.0.0.1:${server.address().port}`})+'\n'));
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>server.close(()=>process.exit(0)));
