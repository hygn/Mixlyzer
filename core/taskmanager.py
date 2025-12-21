from dataclasses import dataclass
from PySide6 import QtGui
import numpy as np
from core.event_bus import EventBus

@dataclass
class task():
    taskid: int
    songname: str
    thumbnail: QtGui.QImage
    status: str
    progress: float

class taskmanager():
    def __init__(self, bus: EventBus):
        self.tasks = {}
        self.taskidlist = []
        self.bus = bus
    def addtask(self, songname:str, thumbnail:QtGui.QImage, status:str, progress:float, send_sig:bool=True) -> task:
        while True:
            tid = np.random.randint(low=0, high=int(2**31-1))
            try: 
                self.taskidlist.index(tid)
            except ValueError:
                self.taskidlist.append(tid)
                break
        tasktoadd = task(
            taskid=tid,
            songname=songname,
            thumbnail=thumbnail,
            status=status,
            progress=progress
        )
        self.tasks.update({tid:tasktoadd})
        if send_sig:
            self.bus.sig_task_created.emit(tid, tasktoadd)
        return tasktoadd
    
    def updatetask(self, tid:int, status:str, progress:float, send_sig:bool=True) -> task:
        self.tasks[tid].status = status
        self.tasks[tid].progress = progress
        if send_sig:
            self.bus.sig_task_progress.emit(tid, progress, status)
        return self.tasks[tid]
    
    def rmtask(self, tid:int, err:str = None, send_sig:bool=True) -> None:
        self.tasks.pop(tid)
        self.taskidlist.remove(tid)
        if send_sig:
            self.bus.sig_task_finished.emit(tid, err)

    def sendInitSignals(self):
        for tid in self.tasks.keys():
            self.bus.sig_task_created.emit(tid, self.tasks[tid])
