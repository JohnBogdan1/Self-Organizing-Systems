import os
import arcade
import random
import math
import time

# --- Constants ---
SPRITE_SCALING_BOID = 0.05
SPRITE_SCALING_OBSTACLE = 0.01
BOID_COUNT = 50
OBSTACLE_COUNT = 10

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Boid Flocking Simulation"

NEARBY_BOID_PERCEPTION = 100
NEARBY_OBSTACLE_PERCEPTION = 100
SEPARATION_DISTANCE = 50

# Counter used to control the FPS
frame_counter = 0


class Boid(arcade.Sprite):
    def __init__(self, filename, scale, id):
        """ Call the parent class. """
        super().__init__(filename, scale)

        self.id = id
        self.speed_x = random.randint(1, 10) / 10.0
        self.speed_y = random.randint(1, 10) / 10.0
        self.max_speed = 1
        # self.angle = random.randint(0, 360)
        self.angle = 0

        self.alignment_factor = 0.01
        self.cohesion_factor = 0.025
        self.separation_factor = 0.5

    def alignment(self, near_boids):
        average_x = 0
        average_y = 0

        nr_boids = len(near_boids)

        if nr_boids == 0:
            return

        for boid in near_boids:
            average_x += (self.center_x - boid.center_x)
            average_y += (self.center_y - boid.center_y)

        average_x /= nr_boids
        average_y /= nr_boids

        self.speed_x += average_x * self.alignment_factor
        self.speed_y += average_y * self.alignment_factor

    def cohesion(self, near_boids):
        average_x = 0
        average_y = 0

        nr_boids = len(near_boids)

        if nr_boids == 0:
            return

        for boid in near_boids:
            average_x += boid.speed_x
            average_y += boid.speed_y

        average_x /= nr_boids
        average_y /= nr_boids

        self.speed_x += average_x * self.cohesion_factor
        self.speed_y += average_y * self.cohesion_factor

    def separation(self, near_boids, near_obstacles):
        nr_boids = len(near_boids)
        nr_obstacles = len(near_obstacles)

        d_x = 0
        d_y = 0

        if nr_boids == 0:
            return

        for boid in near_boids:
            d = arcade.get_distance_between_sprites(self, boid)

            if d < SEPARATION_DISTANCE:
                offset_x = (self.center_x - boid.center_x)
                offset_y = (self.center_y - boid.center_y)
                offset_x = SEPARATION_DISTANCE - offset_x if offset_x > 0 else -SEPARATION_DISTANCE - offset_x
                offset_y = SEPARATION_DISTANCE - offset_y if offset_y > 0 else -SEPARATION_DISTANCE - offset_y
                d_x += offset_x
                d_y += offset_y

        d_x /= nr_boids
        d_y /= nr_boids

        self.speed_x -= d_x * self.separation_factor
        self.speed_y -= d_y * self.separation_factor

        d_x = 0
        d_y = 0

        if nr_obstacles == 0:
            return

        for obstacle in near_obstacles:
            d = arcade.get_distance_between_sprites(self, obstacle)

            if d < SEPARATION_DISTANCE:
                offset_x = (self.center_x - obstacle.center_x)
                offset_y = (self.center_y - obstacle.center_y)
                offset_x = SEPARATION_DISTANCE - offset_x if offset_x > 0 else -SEPARATION_DISTANCE - offset_x
                offset_y = SEPARATION_DISTANCE - offset_y if offset_y > 0 else -SEPARATION_DISTANCE - offset_y
                d_x += offset_x
                d_y += offset_y

        d_x /= nr_obstacles
        d_y /= nr_obstacles

        self.speed_x -= d_x * self.separation_factor
        self.speed_y -= d_y * self.separation_factor

    def edges(self):
        radius = max(self.width, self.height)
        boid_radius = int(radius / 2)

        seed = int(time.time())
        random.seed()

        # better change the sign on velocity
        if self.position[0] + boid_radius > SCREEN_WIDTH and self.speed_x > 0:
            # self.position[0] = boid_radius
            self.speed_x = -self.speed_x
        if self.position[0] - boid_radius < 0 and self.speed_x < 0:
            # self.position[0] = SCREEN_WIDTH - boid_radius
            self.speed_x = -self.speed_x
        if self.position[1] + boid_radius > SCREEN_HEIGHT and self.speed_y > 0:
            # self.position[1] = boid_radius
            self.speed_y = -self.speed_y
        if self.position[1] - boid_radius < 0 and self.speed_y < 0:
            # self.position[1] = SCREEN_HEIGHT - boid_radius
            self.speed_y = -self.speed_y

    def update(self):
        """
        self.change_x = -math.sin(math.radians(self.angle)) * self.speed
        self.change_y = math.cos(math.radians(self.angle)) * self.speed

        self.center_x += self.change_x
        self.center_y += self.change_y
        """
        self.change_x = self.speed_x
        self.change_y = self.speed_y

        if abs(self.speed_x) > self.max_speed or abs(self.speed_y) > self.max_speed:
            speed_factor = self.max_speed / max(abs(self.speed_x), abs(self.speed_y))
            self.change_x *= speed_factor
            self.change_y *= speed_factor

        self.center_x += self.change_x
        self.center_y += self.change_y

        self.angle = math.degrees(math.atan2(self.change_y, self.change_x))

        """ Call the parent class. """
        super().update()


class Obstacle(arcade.Sprite):
    def __init__(self, filename, scale, id):
        """ Call the parent class. """
        super().__init__(filename, scale)

        self.id = id

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.BLACK)

        # Set the working directory
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        # Variables that will hold sprite lists
        self.boid_list = None
        self.obstacles_list = None

    def setup(self):
        # Set up your game here

        # Sprite lists
        self.boid_list = arcade.SpriteList()
        self.obstacles_list = arcade.SpriteList()

        attempts = 10 * OBSTACLE_COUNT * OBSTACLE_COUNT
        generated = 0

        # Random obstacle initialization
        while attempts > 0 and generated < OBSTACLE_COUNT:
            obstacle = Obstacle("images/obstacle.png", SPRITE_SCALING_OBSTACLE, generated)

            radius = max(obstacle.width, obstacle.height)
            obstacle_radius = int(radius / 2)

            # Position the obstacle
            obstacle.center_x = random.randrange(obstacle_radius, SCREEN_WIDTH - obstacle_radius)
            obstacle.center_y = random.randrange(obstacle_radius, SCREEN_HEIGHT - obstacle_radius)

            good_pos = True

            # avoid to place in an occupied zone
            for created_obstacle in self.obstacles_list:
                if arcade.get_distance_between_sprites(created_obstacle, obstacle) <= radius:
                    good_pos = False

            if good_pos:
                generated += 1
                # Add the obstacle to the list
                self.obstacles_list.append(obstacle)

        attempts = 10 * BOID_COUNT * BOID_COUNT
        generated = 0

        # Random boid initialization
        while attempts > 0 and generated < BOID_COUNT:
            boid = Boid("images/boid.png", SPRITE_SCALING_BOID, generated)

            radius = max(boid.width, boid.height)
            boid_radius = int(radius / 2)

            # Position the boid
            boid.center_x = random.randrange(boid_radius, SCREEN_WIDTH - boid_radius)
            boid.center_y = random.randrange(boid_radius, SCREEN_HEIGHT - boid_radius)

            good_pos = True

            # avoid to place in an occupied zone
            for created_boid in self.boid_list:
                if arcade.get_distance_between_sprites(created_boid, boid) <= radius:
                    good_pos = False

            for created_obstacle in self.obstacles_list:
                if arcade.get_distance_between_sprites(boid, created_obstacle) <= radius:
                    good_pos = False

            if good_pos:
                generated += 1
                # Add the boid to the list
                self.boid_list.append(boid)

    def on_draw(self):
        """ Render the screen. """
        self.clear()

        self.boid_list.draw()
        self.obstacles_list.draw()

    def on_update(self, delta_time):
        """ All the logic to move, and the game logic goes here. """
        global frame_counter

        frame_counter += 1

        # Do action on lower FPS
        if frame_counter == 1:
            # self.boid_list.update()
            self.obstacles_list.update()

            for boid in self.boid_list:
                near_boids = []
                near_obstacles = []

                for near_boid in self.boid_list:
                    if boid.id == near_boid.id:
                        continue

                    if arcade.get_distance_between_sprites(boid, near_boid) <= NEARBY_BOID_PERCEPTION:
                        near_boids.append(near_boid)

                for near_obstacle in self.obstacles_list:
                    if arcade.get_distance_between_sprites(boid, near_obstacle) <= NEARBY_OBSTACLE_PERCEPTION:
                        near_obstacles.append(near_obstacle)

                boid.alignment(near_boids)
                boid.cohesion(near_boids)
                boid.separation(near_boids, near_obstacles)

                boid.edges()
                boid.update()

            frame_counter = 0


def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
