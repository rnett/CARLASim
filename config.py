import warnings
from enum import Enum

import carla


class Rain(Enum):
    Clear = 1
    Cloudy = 2
    Wet = 3
    WetCloudy = 4
    Soft = 5
    Mid = 6
    Hard = 7


class Weather:
    def __init__(self, rain: Rain, sunset: bool = False):
        self.rain = rain
        self.sunset = sunset

        try:

            if rain is Rain.Clear:
                if sunset:
                    self.params = carla.WeatherParameters.ClearSunset
                else:
                    self.params = carla.WeatherParameters.ClearNoon
            elif rain is Rain.Cloudy:
                if sunset:
                    self.params = carla.WeatherParameters.CloudySunset
                else:
                    self.params = carla.WeatherParameters.CloudyNoon
            elif rain is Rain.Wet:
                if sunset:
                    self.params = carla.WeatherParameters.WetSunset
                else:
                    self.params = carla.WeatherParameters.WetNoon
            elif rain is Rain.WetCloudy:
                if sunset:
                    self.params = carla.WeatherParameters.WetCloudySunset
                else:
                    self.params = carla.WeatherParameters.WetCloudyNoon
            elif rain is Rain.Soft:
                if sunset:
                    self.params = carla.WeatherParameters.SoftRainSunset
                else:
                    self.params = carla.WeatherParameters.SoftRainNoon
            elif rain is Rain.Mid:
                if sunset:
                    self.params = carla.WeatherParameters.MidRainSunset
                else:
                    self.params = carla.WeatherParameters.MidRainyNoon
            else:  # Rain.Hard
                if sunset:
                    self.params = carla.WeatherParameters.HardRainSunset
                else:
                    self.params = carla.WeatherParameters.HardRainNoon

        except AttributeError as ae:
            if str(ae) == "module 'carla' has no attribute 'WeatherParameters'":
                warnings.warn("Can't load carla WeatherParameters, skiping",
                              ImportWarning)
            else:
                raise ae


class City(Enum):
    Town01 = 1
    Town02 = 2
    Town03 = 3
    Town04 = 4
    Town05 = 5

    @property
    def map(self) -> str:
        return f"/Game/Carla/Maps/{self.name}"


class SimConfig:

    def __init__(self, cars: int, pedestrians: int,
                 city: City = City.Town01,
                 rain: Rain = Rain.Clear,
                 sunset: bool = False):

        self.pedestrians = pedestrians
        self.cars = cars
        self.sunset = sunset
        self.rain = rain
        self.city = city
        self.weather = Weather(rain, sunset)

    @property
    def folder_name(self) -> str:
        if self.sunset:
            time = 'sunset'
        else:
            time = 'noon'

        return f"{self.city.name.lower()}/" \
               f"{self.rain.name.lower()}/{time}/cars_{self.cars}_peds_" \
               f"{self.pedestrians}"

    @staticmethod
    def from_folder_name(name: str):
        """
        name should be like "[/]{town}/{rain}/{time}/{folder_name}[/]"
        """

        parts = name.strip('/').split('/')

        town_name = parts[0].capitalize()

        town = None

        for t in list(City):
            if t.name == town_name:
                town = t

        if town is None:
            raise ValueError(f"No town found for name {parts[0]}")

        rain_name = parts[1].capitalize()

        rain = None

        for t in list(Rain):
            if t.name == rain_name:
                rain = t

        if rain is None:
            raise ValueError(f"No rain setting found for name {parts[1]}")

        sunset = parts[2] == "sunset"

        folder_name = parts[3].split('_')
        cars = int(folder_name[1])
        peds = int(folder_name[3])
        return SimConfig(cars, peds, town, rain, sunset)
