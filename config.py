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


def _weather_params(cloudyness: int, precipitation: int, precipitation_deposits: int, wind_intensity: float,
                    sunset: bool):
    if sunset:
        sun_alt_angel = 15
    else:
        sun_alt_angel = 75

    return carla.WeatherParameters(cloudyness, precipitation, precipitation_deposits, wind_intensity, 0, sun_alt_angel)


class Weather:
    def __init__(self, rain: Rain, sunset: bool = False):
        self.rain = rain
        self.sunset = sunset

        try:
            if rain is Rain.Clear:
                self.params = _weather_params(1, 0, 0, 0.2, sunset)
            elif rain is Rain.Cloudy:
                self.params = _weather_params(80, 0, 0, 0.35, sunset)
            elif rain is Rain.Wet:
                self.params = _weather_params(10, 0, 50, 0.35, sunset)
            elif rain is Rain.WetCloudy:
                self.params = _weather_params(80, 0, 60, 0.35, sunset)
            elif rain is Rain.Soft:
                self.params = _weather_params(70, 15, 60, 0.35, sunset)
            elif rain is Rain.Mid:
                self.params = _weather_params(80, 55, 70, 0.5, sunset)
            else:  # Rain.Hard
                self.params = _weather_params(90, 85, 100, 1, sunset)

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

        town_name = parts[0]

        town = None

        for t in list(City):
            if t.name.lower() == town_name.lower():
                town = t
                break

        if town is None:
            raise ValueError(f"No town found for name {parts[0]}")

        rain_name = parts[1]

        rain = None

        for t in list(Rain):
            if t.name.lower() == rain_name.lower():
                rain = t
                break

        if rain is None:
            raise ValueError(f"No rain setting found for name {parts[1]}")

        sunset = parts[2] == "sunset"

        folder_name = parts[3].split('_')
        cars = int(folder_name[1])
        peds = int(folder_name[3])
        return SimConfig(cars, peds, town, rain, sunset)
