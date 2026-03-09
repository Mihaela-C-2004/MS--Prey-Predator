import pygame
import random
import math
import matplotlib.pyplot as plt
from collections import deque

pygame.init()

WIDTH, HEIGHT = 1000, 650

BACKGROUND_COLOR = (20, 20, 30)
PREY_COLOR = (0, 200, 80)
PREDATOR_COLOR = (220, 40, 40)
TEXT_COLOR = (200, 200, 200)
FOOD_COLOR = (200, 180, 30)
OBSTACLE_COLOR = (100, 100, 110)
TRAIL_COLOR = (120, 120, 120)

FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation - Extended")
clock = pygame.time.Clock()

FONT = pygame.font.SysFont(None, 20)

# Global params
GLOBAL_ENERGY_CONSUMPTION = 0.01  # energy lost per tick
REPRODUCTION_ENERGY_THRESHOLD = 80.0  # energy required to attempt reproduction
REPRODUCTION_COOLDOWN = 3000  # ms minimum between reproductions per agent
REPRODUCTION_WAIT_FRAMES = 30  # frames agents wait together before birth
MAX_FOOD = 500
MAX_OBSTACLES = 100
SHOW_TRAILS = False

# energy consumption per frame
def clamp(v, a, b):
    return max(a, min(v, b))
# random position within screen with margin
def rnd_pos(margin=20):
    return pygame.math.Vector2(random.uniform(margin, WIDTH-margin), random.uniform(margin, HEIGHT-margin))

class Food:
    def __init__(self, position=None, energy_value=40):
        self.position = position or rnd_pos()
        self.energy_value = energy_value
        self.size = 4

    def draw(self):
        rect = pygame.Rect(int(self.position.x - self.size/2), int(self.position.y - self.size/2), self.size, self.size)
        pygame.draw.rect(screen, FOOD_COLOR, rect)

class Obstacle:
    def __init__(self, position=None, radius=12): 
        self.position = position or rnd_pos()
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, OBSTACLE_COLOR, (int(self.position.x), int(self.position.y)), self.radius)
        pygame.draw.circle(screen, (60,60,70), (int(self.position.x), int(self.position.y)), self.radius, 2)

class Agent:
    def __init__(self, position=None, velocity=None, base_speed=2.0, color=(255,255,255), energy=50.0):
        self.position = position or rnd_pos()
        v = velocity or pygame.math.Vector2(random.uniform(-1,1), random.uniform(-1,1))
        if v.length_squared() == 0:
            v = pygame.math.Vector2(1, 0)
        self.velocity = v.normalize()
        self.base_speed = base_speed
        self.color = color
        self.trail = deque(maxlen=25)
        self.energy = energy
        self.last_reproduction = -99999 
        self.reproduction_wait_counter = 0
        self.reproduction_partner = None

    def update_position(self):
        speed = self.current_speed()
        self.position += self.velocity * speed
        # bounce off walls
        if self.position.x < 0:
            self.position.x = 0
            self.velocity.x *= -1
        if self.position.x > WIDTH:
            self.position.x = WIDTH
            self.velocity.x *= -1
        if self.position.y < 0:
            self.position.y = 0
            self.velocity.y *= -1
        if self.position.y > HEIGHT:
            self.position.y = HEIGHT
            self.velocity.y *= -1
        self.trail.append(self.position.copy())

    def current_speed(self):
        return self.base_speed

    def draw_trail(self):
        if SHOW_TRAILS and len(self.trail) > 1:
            pts = [(int(p.x), int(p.y)) for p in self.trail]
            pygame.draw.lines(screen, TRAIL_COLOR, False, pts, 1)

    def draw_energy_bar(self, offset_y=-12, width=20):
        pct = clamp(self.energy / 100.0, 0.0, 1.0)
        w = width
        h = 4
        x = int(self.position.x - w/2)
        y = int(self.position.y + offset_y)
        bg = pygame.Rect(x, y, w, h)
        fg = pygame.Rect(x, y, int(w*pct), h)
        pygame.draw.rect(screen, (50,50,50), bg)
        color = (50 + int(205*(1-pct)), 50 + int(205*pct), 30)
        pygame.draw.rect(screen, color, fg)


class Prey(Agent):
    def __init__(self):
        super().__init__(base_speed=1.8, color=PREY_COLOR, energy=random.uniform(80, 100))
        self.vision_radius = 90
        self.separation_radius = 16
        self.max_trail_length = 25

    def current_speed(self):
        # speed scales slightly with flock size
        return self.base_speed * (1.0 + getattr(self, 'flock_speed_bonus', 0.0))

    def update(self, predators, prey_list, foods, obstacles, dt, current_time, flocking_on=True):
        # energy consumption
        self.energy -= GLOBAL_ENERGY_CONSUMPTION * dt * 0.016  # scaled by frame-time

        # AI priorities:
        # 1) flee predator if close
        # 2) seek food if low energy
        # 3) flocking with nearby prey
        # 4) wander

        flee_vec = pygame.math.Vector2(0,0)
        nearest_pred = None
        minpd = float('inf')
        for pred in predators:
            d = self.position.distance_to(pred.position)
            if d < minpd:
                minpd = d
                nearest_pred = pred
        if nearest_pred and minpd < self.vision_radius:
            # stronger flee when closer
            flee_vec = (self.position - nearest_pred.position)
            if flee_vec.length_squared() > 0:
                flee_vec = flee_vec.normalize() * (1.5 + (self.vision_radius - minpd)/self.vision_radius)

        # food seeking
        want_food = self.energy < 70 
        food_vec = pygame.math.Vector2(0,0)
        if want_food and foods:
            nearest_food = None
            minfd = float('inf')
            for f in foods:
                d = self.position.distance_to(f.position)
                if d < minfd:
                    minfd = d
                    nearest_food = f
            if nearest_food and minfd < 200:
                food_vec = (nearest_food.position - self.position)
                if food_vec.length_squared() > 0:
                    food_vec = food_vec.normalize() * (1.0 + (200 - minfd)/200)

        # flocking behaviors (boids) - only affect direction, not absolute speed
        sep = pygame.math.Vector2(0,0)
        ali = pygame.math.Vector2(0,0)
        coh = pygame.math.Vector2(0,0)
        neighbor_count = 0
        for other in prey_list:
            if other is self: continue
            d = self.position.distance_to(other.position)
            if d < 1e-5: continue #ignores the exact same position
            if d < 80:
                neighbor_count += 1
                # separation
                if d < self.separation_radius:
                    sep += (self.position - other.position) / d
                # alignment
                ali += other.velocity
                # cohesion
                coh += other.position

        if neighbor_count > 0:
            ali /= neighbor_count
            coh /= neighbor_count
            coh = (coh - self.position)
            # normalize
            if ali.length_squared() > 0:
                ali = ali.normalize()
            if coh.length_squared() > 0:
                coh = coh.normalize()

        # combine behavior vectors: weight them
        steer = pygame.math.Vector2(0,0)
        if flee_vec.length_squared() > 0:
            steer += flee_vec * 2.5
        if food_vec.length_squared() > 0:
            steer += food_vec * 1.8
        if flocking_on:
            steer += sep * 2.0 + ali * 0.7 + coh * 0.6

            # small speed bonus with flock size
            self.flock_speed_bonus = min(0.8, neighbor_count / 30.0)
        else:
            self.flock_speed_bonus = 0.0

        # obstacle avoidance
        avoid = pygame.math.Vector2(0,0)
        for obs in obstacles:
            d = self.position.distance_to(obs.position)
            if d < obs.radius + 40:
                diff = self.position - obs.position
                if diff.length_squared() > 0:
                    avoid += diff.normalize() * (1.0 + (obs.radius + 40 - d)/ (obs.radius + 40))
        steer += avoid * 2.0

        # slight wandering when no strong steer (jitter)
        if steer.length_squared() < 0.01:
            jitter = pygame.math.Vector2(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
            self.velocity = (self.velocity + jitter).normalize()
        else:  # apply steering, guiding velocity
            new_dir = (self.velocity + steer).normalize()
            self.velocity = new_dir

        # reproduction attempt: if energy high and close to another high-energy prey
        baby = None
        if self.energy > REPRODUCTION_ENERGY_THRESHOLD:
            # find partner
            for other in prey_list:
                if other is self: continue
                if other.energy > REPRODUCTION_ENERGY_THRESHOLD:
                    d = self.position.distance_to(other.position)
                    if d < 10:
                        # check cooldowns
                        if (current_time - self.last_reproduction > REPRODUCTION_COOLDOWN and
                                current_time - other.last_reproduction > REPRODUCTION_COOLDOWN):
                            # pair up: increment wait counters (set for both)
                            if self.reproduction_partner is None:
                                self.reproduction_partner = other
                                self.reproduction_wait_counter = REPRODUCTION_WAIT_FRAMES
                                other.reproduction_partner = self
                                other.reproduction_wait_counter = REPRODUCTION_WAIT_FRAMES
                        break

        # if in reproduction wait: both stop moving and count down
        if self.reproduction_partner is not None and self.reproduction_wait_counter > 0:
            # zero velocity (stop)
            self.reproduction_wait_counter -= 1
            # don't move much while waiting (keep a tiny jiggle), if partner disappeared or energy went down, cancel
            partner = self.reproduction_partner
            if partner is None or partner.energy < REPRODUCTION_ENERGY_THRESHOLD - 10:
                self.reproduction_partner = None
                self.reproduction_wait_counter = 0
            else:
                # stay near partner
                direction_to_partner = (partner.position - self.position)
                if direction_to_partner.length_squared() > 1:
                    self.velocity = direction_to_partner.normalize() * 0.1

            # when counter reached zero, perform birth (only one parent creates object)
            if self.reproduction_wait_counter == 0:
                partner = self.reproduction_partner
                if partner and partner.reproduction_partner is self and partner.reproduction_wait_counter == 0:
                    if (current_time - self.last_reproduction > REPRODUCTION_COOLDOWN and
                        current_time - partner.last_reproduction > REPRODUCTION_COOLDOWN):
                        # ensure only one parent creates the baby
                        if id(self) < id(partner):
                            mid = (self.position + partner.position) / 2 + pygame.math.Vector2(random.uniform(-6,6), random.uniform(-6,6))
                            baby = Prey()
                            baby.position = mid
                            baby.energy = 60.0
                            # update parents' states and energies
                            self.last_reproduction = current_time
                            partner.last_reproduction = current_time
                            self.energy = clamp(self.energy - 20, 0, 200)
                            partner.energy = clamp(partner.energy - 20, 0, 200)
                    # clear pairing for both sides
                    if partner:
                        partner.reproduction_partner = None
                        partner.reproduction_wait_counter = 0
                    self.reproduction_partner = None
                    self.reproduction_wait_counter = 0
        else:
            pass

        self.update_position()

        # check for food consumption (if near food)
        eaten = None
        for f in foods:
            if self.position.distance_to(f.position) < 8:
                self.energy = clamp(self.energy + f.energy_value, 0, 150)
                eaten = f
                break
        if eaten:
            foods.remove(eaten)

        return baby

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), 5)
        self.draw_trail()
        self.draw_energy_bar()


class Predator(Agent):
    def __init__(self):
        super().__init__(base_speed=2.4, color=PREDATOR_COLOR, energy=random.uniform(50, 120))
        self.vision_radius = 140
        self.max_trail_length = 25

    def current_speed(self):
        return self.base_speed * (1.0 + getattr(self, 'hunger_speed_bonus', 0.0))

    def update(self, prey_list, predator_list, obstacles, dt, current_time):  # AI for predator
        self.energy -= GLOBAL_ENERGY_CONSUMPTION * 1.2 * dt * 0.016

        if prey_list:
            nearest = None
            mind = float('inf')
            for p in prey_list:
                d = self.position.distance_to(p.position)
                if d < mind:
                    mind = d
                    nearest = p
            if nearest and mind < self.vision_radius:
                # hunt: move toward prey
                dir_vec = (nearest.position - self.position)
                if dir_vec.length_squared() > 0:
                    self.velocity = dir_vec.normalize()
            else:
                # wander / random slightly 
                jitter = pygame.math.Vector2(random.uniform(-0.2,0.2), random.uniform(-0.2,0.2))
                self.velocity = (self.velocity + jitter).normalize()
        else: #no prey 
            jitter = pygame.math.Vector2(random.uniform(-0.2,0.2), random.uniform(-0.2,0.2))
            self.velocity = (self.velocity + jitter).normalize()

        # obstacle avoidance
        avoid = pygame.math.Vector2(0,0)
        for obs in obstacles:
            d = self.position.distance_to(obs.position)
            if d < obs.radius + 50:
                diff = self.position - obs.position
                if diff.length_squared() > 0:
                    avoid += diff.normalize() * (1.0 + (obs.radius + 50 - d)/ (obs.radius + 50))
        self.velocity = (self.velocity + avoid*1.2).normalize()

        # small speed bonus depending on hunger (hungry predators a bit faster)
        self.hunger_speed_bonus = clamp((120 - self.energy) / 200.0, -0.2, 0.6)

        # reproduction attempt
        # returns a baby Predator instance when reproduction occurs, otherwise None
        baby = None

        if self.energy > REPRODUCTION_ENERGY_THRESHOLD + 20:
            for other in predator_list:
                if other is self:
                    continue
        

                d = self.position.distance_to(other.position)
                if d < 8 and other.energy > REPRODUCTION_ENERGY_THRESHOLD + 20:

                    if (current_time - self.last_reproduction > REPRODUCTION_COOLDOWN and
                        current_time - other.last_reproduction > REPRODUCTION_COOLDOWN):
                        # Create baby predator
                        mid = (self.position + other.position) / 2 + pygame.Vector2(random.uniform(-6, 6), random.uniform(-6, 6))

                        baby = Predator()
                        baby.position = mid
                        baby.energy = 50.0

                        # Update parents
                        self.last_reproduction = current_time
                        other.last_reproduction = current_time

                        self.energy = clamp(self.energy - 40, 0, 300)
                        other.energy = clamp(other.energy - 40, 0, 300)

                        break

        self.update_position()
        return baby


    def draw(self):
        # draw as triangle rotated
        angle = self.velocity.angle_to(pygame.math.Vector2(1, 0))
        point_list = [
            pygame.math.Vector2(12, 0),
            pygame.math.Vector2(-6, -6),
            pygame.math.Vector2(-6, 6),
        ]
        rotated_points = [self.position + p.rotate(-angle) for p in point_list]
        pygame.draw.polygon(screen, self.color, rotated_points)
        self.draw_trail()
        self.draw_energy_bar(offset_y=-16)

class Simulation:
    def __init__(self, num_prey=100, num_predators=3):
        self.prey_list = [Prey() for _ in range(num_prey)]
        self.predator_list = [Predator() for _ in range(num_predators)]
        self.foods = [Food() for _ in range(200)]
        # create more smaller obstacles at start
        initial_obs = 10
        self.obstacles = [Obstacle(position=rnd_pos(40), radius=random.randint(10,20)) for _ in range(initial_obs)]
        self.running = True
        self.spawn_counts = {'prey':0, 'predator':0}
        self.history = {
            'time': [],
            'prey_count': [],
            'predator_count': [],
            'food_count': [],
            'prey_births': [],
            'predator_births': []
        }
        self.last_time = pygame.time.get_ticks()
        self.flocking_on = True
        self.reproduction_rate = REPRODUCTION_ENERGY_THRESHOLD
        self.reproduction_cooldown = REPRODUCTION_COOLDOWN
        self.global_energy_consumption = GLOBAL_ENERGY_CONSUMPTION
        self.prey_births_this_step = 0
        self.predator_births_this_step = 0

    def run(self):
        while self.running:
            dt = clock.tick(FPS)
            current_time = pygame.time.get_ticks()
            self.handle_events()
            self.update_agents(dt, current_time)
            self.handle_collisions()
            self.render()
            self.record_history(current_time)

        pygame.quit()

    def handle_events(self):
        global GLOBAL_ENERGY_CONSUMPTION, REPRODUCTION_ENERGY_THRESHOLD, REPRODUCTION_COOLDOWN, SHOW_TRAILS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.add_prey()
                elif event.key == pygame.K_o:
                    self.add_predator()
                elif event.key == pygame.K_f:
                    self.flocking_on = not self.flocking_on
                elif event.key == pygame.K_LEFTBRACKET:
                    # reduce reproduction threshold => easier to reproduce
                    REPRODUCTION_ENERGY_THRESHOLD = max(40, REPRODUCTION_ENERGY_THRESHOLD - 5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    REPRODUCTION_ENERGY_THRESHOLD = min(150, REPRODUCTION_ENERGY_THRESHOLD + 5)
                elif event.key == pygame.K_MINUS:
                    GLOBAL_ENERGY_CONSUMPTION = max(0.001, GLOBAL_ENERGY_CONSUMPTION - 0.002)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    GLOBAL_ENERGY_CONSUMPTION = min(0.1, GLOBAL_ENERGY_CONSUMPTION + 0.002)
                elif event.key == pygame.K_g:
                    # show graphs
                    self.show_graphs()
                elif event.key == pygame.K_t:
                    SHOW_TRAILS = not SHOW_TRAILS
                elif event.key == pygame.K_c:
                    # clear foods and obstacles
                    self.foods.clear()
                    self.obstacles.clear()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click add food
                    pos = pygame.math.Vector2(*event.pos)
                    self.foods.append(Food(position=pos))
                elif event.button == 3:  # right click add obstacle
                    pos = pygame.math.Vector2(*event.pos)
                    self.obstacles.append(Obstacle(position=pos, radius=8 + random.randint(2,14)))

    def add_prey(self):
        self.prey_list.append(Prey())

    def add_predator(self):
        self.predator_list.append(Predator())

    def update_agents(self, dt, current_time):
        # reset births counter
        self.prey_births_this_step = 0
        self.predator_births_this_step = 0

        # Update prey
        prey_babies = []
        for prey in list(self.prey_list):
            baby = prey.update(self.predator_list, self.prey_list, self.foods, self.obstacles, dt, current_time, flocking_on=self.flocking_on)
            if baby:
                prey_babies.append(baby)

        # add prey babies produced this frame
        for baby in prey_babies:
            self.prey_list.append(baby)
            self.prey_births_this_step += 1

        # Update predators and collect babies produced by predators
        babies = []
        for pred in list(self.predator_list):
            baby = pred.update(self.prey_list, self.predator_list, self.obstacles, dt, current_time)
            if baby:
                babies.append(baby)

        for baby in babies:
            self.predator_list.append(baby)
            self.predator_births_this_step += 1

        # Predators consume prey if close
        for pred in list(self.predator_list):
            for prey in list(self.prey_list):
                if pred.position.distance_to(prey.position) < 8:
                    pred.energy = clamp(pred.energy + 30, 0, 300)
                    try:
                        self.prey_list.remove(prey)
                    except ValueError:
                        pass
                    break

        # remove dead agents (energy <= 0)
        self.prey_list = [p for p in self.prey_list if p.energy > 0]
        self.predator_list = [p for p in self.predator_list if p.energy > 0]

        # keep food and obstacles counts reasonable
        if len(self.foods) < 100 and random.random() < 0.06:
            for _ in range(random.randint(5, 10)):
                self.foods.append(Food())
        if len(self.foods) > MAX_FOOD:
            self.foods = self.foods[-MAX_FOOD:]
        if len(self.obstacles) > MAX_OBSTACLES:
            self.obstacles = self.obstacles[-MAX_OBSTACLES:]


    def handle_collisions(self):
        pass

    def render(self):
        screen.fill(BACKGROUND_COLOR)
        # draw obstacles
        for obs in self.obstacles:
            obs.draw()
        # draw food
        for f in self.foods:
            f.draw()
        # draw prey
        for prey in self.prey_list:
            prey.draw()
        # draw predators
        for pred in self.predator_list:
            pred.draw()

        self.draw_legend()
        self.draw_stats()
        pygame.display.flip()

    def draw_legend(self):
        lines = [
            "Controls: Left click = add food | Right click = add obstacle | P = add prey | O = add predator",
            "F = toggle flocking | [ or ] = reproduction threshold | - or = for energy consumption | G = show graphs | T toggle trails",
        ]
        y = 8
        for ln in lines:
            surf = FONT.render(ln, True, TEXT_COLOR)
            screen.blit(surf, (10, y))
            y += 18

    def draw_stats(self):
        y = HEIGHT - 130
        stats = [
            f"Prey: {len(self.prey_list)}",
            f"Predators: {len(self.predator_list)}",
            f"Food: {len(self.foods)}",
            f"Flocking: {'On' if self.flocking_on else 'Off'}",
            f"Reprod threshold: {REPRODUCTION_ENERGY_THRESHOLD}",
            f"Energy consumption: {GLOBAL_ENERGY_CONSUMPTION:.3f}",
        ]
        x = WIDTH - 260
        for i, s in enumerate(stats):
            surf = FONT.render(s, True, TEXT_COLOR)
            screen.blit(surf, (x, y + i*18))

    def record_history(self, current_time):
        self.history['time'].append(pygame.time.get_ticks()/1000.0)
        self.history['prey_count'].append(len(self.prey_list))
        self.history['predator_count'].append(len(self.predator_list))
        self.history['food_count'].append(len(self.foods))
        self.history['prey_births'].append(self.prey_births_this_step)
        self.history['predator_births'].append(self.predator_births_this_step)

    def show_graphs(self):
        t = self.history['time']
        if not t:
            print("No history yet.")
            return
        prey = self.history['prey_count']
        pred = self.history['predator_count']
        food = self.history['food_count']
        pb = self.history['prey_births']
        db = self.history['predator_births']

        # Plot 1: population over time
        plt.figure(figsize=(10,6))
        plt.plot(t, prey, label='Prey count')
        plt.plot(t, pred, label='Predator count')
        plt.plot(t, food, label='Food count')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.title('Populations over time')
        plt.legend()
        plt.grid(True)

        # Plot 2: births per timestep over time
        plt.figure(figsize=(10,4))
        plt.plot(t, pb, label='Prey births per frame')
        plt.plot(t, db, label='Predator births per frame')
        plt.xlabel('Time (s)')
        plt.ylabel('Births')
        plt.title('Birth rates over time')
        plt.legend()
        plt.grid(True)

        # Plot 3: phase space (prey vs predator)
        plt.figure(figsize=(6,6))
        plt.scatter(prey, pred, s=6, alpha=0.6)
        plt.xlabel('Prey count')
        plt.ylabel('Predator count')
        plt.title('Phase space: Prey vs Predator')

        plt.show()

if __name__ == "__main__":
    sim = Simulation(num_prey=100, num_predators=3)
    sim.run()