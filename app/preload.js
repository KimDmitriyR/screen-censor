const { contextBridge, ipcRenderer } = require('electron');

console.log("PRELOAD LOADED");

contextBridge.exposeInMainWorld('electronAPI', {
  getSources: () => ipcRenderer.invoke('get-sources')
});