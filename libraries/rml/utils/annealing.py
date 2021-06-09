import matplotlib.pyplot as plt

class CyclicalLr:
    def __init__(self, schedule, half_cycle_length: int):
        self.schedule = schedule
        self.half_cycle_length = half_cycle_length
        self.forward = True
    
    def step(self, epoch = None):
        if epoch is not None:
            raise NotImplementedError
        
        inc = 1 if self.forward else -1
        
        self.schedule.step(self.schedule.last_epoch + inc)
        self.update_direction()
        
    def update_direction(self):
        if self.schedule.last_epoch == self.half_cycle_length:
            self.forward = False
        
        if self.schedule.last_epoch == 0:
            self.forward = True
    
class RestartsLr:
    def __init__(self, schedule_fn, step_size: int, step_size_multiplier: int = 1):
        self.schedule = schedule_fn(step_size)
        self.step_size = step_size
        self.schedule_fn = schedule_fn
        self.step_size_multiplier = step_size_multiplier
    
    def step(self, epoch = None):
        if epoch is not None:
            raise NotImplementedError
                
        epoch = self.schedule.last_epoch + 1
        
        if epoch > self.step_size:
            epoch = 0
            self.step_size = self.step_size * self.step_size_multiplier
            self.schedule = self.schedule_fn(self.step_size)
            
        self.schedule.step(epoch)
        

def get_lr(optim) -> float:
    return optim.state_dict()["param_groups"][0]["lr"]


def test_scheduel(schedual, optim, ittrs):
    lrs = []

    for i in range(ittrs):
        schedual.step()
        lrs.append(get_lr(optim))

    plt.semilogy(range(len(lrs)), lrs)
    
