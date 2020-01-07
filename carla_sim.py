import logging
import os
import random
import shutil

import sys
from pathlib import Path

sys.path.append('~/carla/CARLA_0.9.6/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')

import imageio

import image_converter
from config import SimConfig
from recordings import Recording
from sides import Side, SideMap

from carla_constants import *

import carla

class CarlaSim:
    def __init__(self,
                 config: SimConfig,
                 base_output_folder: str = '/data/carla/',
                 overwrite: bool = False,
                 seed=123,
                 car_idx=None,
                 host='localhost', port='2000'):

        random.seed(seed)

        self.config = config
        self.num_peds = config.pedestrians
        self.num_cars = config.cars
        self.host = host
        self.port = port
        self.rgb_cameras = SideMap()
        self.depth_cameras = SideMap()
        self.ticks = 0

        self.output_folder = base_output_folder + self.config.folder_name + '/'

        self.recording = Recording(base_output_folder, config)

        if os.path.exists(self.output_folder):
            if overwrite:
                shutil.rmtree(self.output_folder)
            else:
                print(
                    f"Output folder {self.output_folder} already exists and "
                    f"'overwrite' is false.")
                raise FileExistsError
        else:
            Path(self.output_folder).mkdir(exist_ok=True, parents=True)

        with open(self.output_folder + "seed.txt", 'x') as seed_file:
            seed_file.write(str(seed))

        try:
            # First of all, we need to create the client that will send the
            # requests
            # to the simulator. Here we'll assume the simulator is accepting
            # requests in the localhost at port 2000.
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(2.0)

            # Once we have a client we can retrieve the world that is currently
            # running.
            # print(self.client.get_available_maps())
            self.world = self.client.load_world(self.config.city.map)

            # The world contains the list blueprints that we can use for
            # adding new
            # actors into the simulation.

            print(f"Spawning {self.num_cars} cars")

            blueprints = self.world.get_blueprint_library().filter("vehicle.*")
            blueprintsWalkers = self.world.get_blueprint_library().filter(
                "walker.pedestrian.*")

            blueprints = [x for x in blueprints if
                          int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if
                          not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if
                          not x.id.endswith('carlacola')]

            self.cars = []

            spawn_points = self.world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if self.num_cars <= number_of_spawn_points:
                random.shuffle(spawn_points)
            else:
                raise ValueError(
                    f"Number of cars {self.num_cars} greater than number of "
                    f"spawn "
                    f"points {number_of_spawn_points}")

            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            batch = []
            for n, transform in enumerate(spawn_points):

                if n >= self.num_cars:
                    break

                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(
                        blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(
                        blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(
                    SetAutopilot(FutureActor, True)))

            for response in self.client.apply_batch_sync(batch):
                if response.error:
                    logging.error(response.error)
                else:
                    # print("spawned car", response.actor_id)
                    self.cars.append(response.actor_id)

            self.cars = [self.world.get_actor(x) for x in self.cars]

            # -----------  WALKERS  -----------

            print(f"Spawning {self.num_peds} walkers")

            spawn_points = []
            for i in range(self.num_peds):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            walkers_list = []
            all_id = []

            # Spawn walkers
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invencible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    # print("spawned walker", results[i].actor_id)
                    walkers_list.append({"id": results[i].actor_id})

            batch = []
            walker_controller_bp = self.world.get_blueprint_library().find(
                'controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(),
                                        walkers_list[i]["id"]))
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id

            # 4. we put altogether the walkers and controllers id to get the
            # objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = self.world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of
            # the walkers we have just created
            self.world.wait_for_tick()

            # 5. initialize each controller and set target to walk to (list
            # is [controler, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())
                # random max speed
                all_actors[i].set_max_speed(
                    1 + random.random())  # max speed between 1 and 2 (
                # default is 1.4 m/s)

            self.pedestrians = [x['id'] for x in walkers_list]
            self.pedestrians = [self.world.get_actor(x) for x in
                                self.pedestrians]

            blueprint_library = self.world.get_blueprint_library()

            # Add main car
            if car_idx is None:
                bp = random.choice(blueprint_library.filter('vehicle'))
            else:
                bp = blueprint_library.filter('vehicle')[car_idx]

            if bp.has_attribute('color'):
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            transform = random.choice(self.world.get_map().get_spawn_points())

            # So let's tell the world to spawn the vehicle.
            vehicle = self.world.spawn_actor(bp, transform)

            print('created %s' % vehicle.type_id)

            vehicle.set_autopilot(True)

            self.car = vehicle
            self.world.get_spectator().set_transform(self.car.get_transform())

            self.add_sensors()
            self.world.set_weather(config.weather.params)

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)

        except Exception as e:
            print("Error starting server")
            raise e

    def add_sensors(self):
        blueprint_library = self.world.get_blueprint_library()

        for side in list(Side):
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
            camera_bp.set_attribute('fov', str(FOV))

            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4),
                                               side.rotation())
            camera = self.world.spawn_actor(camera_bp, camera_transform,
                                            attach_to=self.car)
            # print('created %s' % camera.type_id)
            self.rgb_cameras[side] = camera

        for side in list(Side):
            camera_bp = blueprint_library.find('sensor.camera.depth')
            camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
            camera_bp.set_attribute('fov', str(FOV))

            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4),
                                               side.rotation())
            camera = self.world.spawn_actor(camera_bp, camera_transform,
                                            attach_to=self.car)
            # print('created %s' % camera.type_id)
            self.depth_cameras[side] = camera

    def tick(self):
        frame = self.world.tick()
        cc = carla.ColorConverter.Depth

        car_loc = self.car.get_transform()
        vec = car_loc.get_forward_vector()
        loc = car_loc.location

        new_loc = carla.Location(loc.x - vec.x * 12, loc.y - vec.y * 12,
                                 loc.z + 3)

        self.world.get_spectator().set_transform(
            carla.Transform(new_loc, car_loc.rotation))

        if self.ticks % 20 == 0:
            images = self.rgb_cameras.pop(frame)
            for k, v in images.items():
                v.save_to_disk(
                    self.output_folder + f"raw/{k.name.lower()}_"
                                         f"{int(self.ticks / 20)}_rgb.png")

        if self.ticks % 20 == 0:
            images = self.depth_cameras.pop(frame)
            for k, v in images.items():
                norm_depth = image_converter.depth_to_array(v)

                imageio.imwrite(
                    self.output_folder + f"raw/{k.name.lower()}_"
                                         f"{int(self.ticks / 20)}_depth.png",
                    (norm_depth * DEPTH_MULTIPLIER).astype('uint16'))

        self.ticks += 1

        return (self.ticks - 1) % 20 == 0

    def end(self):
        for a in self.cars:
            a.destroy()
        for a in self.pedestrians:
            a.destroy()

        for a in self.rgb_cameras.sides.values():
            a[0].destroy()
        for a in self.depth_cameras.sides.values():
            a[0].destroy()

        self.car.destroy()
