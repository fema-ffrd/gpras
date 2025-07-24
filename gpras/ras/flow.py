"""Utilities to read, edit, and write HEC-RAS unsteady flow files."""

from collections.abc import Callable
from typing import Any, Literal, Self, cast

ICTYPE = Literal["2D", "IC Point"]

PRECIP_MODE = Literal["Enable", "Disable"]
WIND_MODE = Literal["No Wind Forces", "Speed/Direction", "Velocity X/Y"]
BC_PRECIP_MODE = Literal["None", "Point", "Gridded", "Constant"]
BC_ET_MODE = Literal["None", "Point", "Constant"]
BC_WIND_MODE = Literal["Point", "Gridded", "Constant"]
BC_AIR_PRESSURE_MODE = Literal["Point", "Gridded", "Constant"]

PRECIP_UNITS = Literal["mm/hr", "in/hr"]
ET_UNITS = Literal["mm/hr", "in/hr"]

PRECIP_INTERPOLATION_METHODS = Literal[
    "Thiessen Polygon", "Inv Distance Sq", "Inv Distance Sq (Restricted)", "Peak Preservation", "Nearest"
]  # Note that nearest is only used during initialization.  Not a valid method for precip
ET_INTERPOLATION_METHODS = Literal["Nearest", "Inv Distance Sq", "Inv Distance Sq (Restricted)"]
WIND_INTERPOLATION_METHODS = Literal["Nearest", "Inv Distance Sq", "Inv Distance Sq (Restricted)"]


class InitialCondition:
    """An initial condition for a HEC-RAS unsteady flow file."""

    def __init__(self, idx: str, elevation: str, ic_type: ICTYPE):
        """Class constructor."""
        self.idx = idx
        self.ic_type = ic_type
        self.elevation = elevation

    def __str__(self) -> str:
        """Single line that will go in the text file."""
        if self.ic_type == "2D":
            return f"Initial Storage Elev={self.idx}    ,{self.elevation}"
        elif self.ic_type == "IC Point":
            return f"IC Point Elev={self.idx}                      ,{self.elevation}"

    @staticmethod
    def _parse_ic_type(key: str) -> ICTYPE:
        """Determine type of initial condition from text."""
        if key == "Initial Storage Elev":
            return "2D"
        elif key == "IC Point Elev":
            return "IC Point"
        else:
            msg = f"Initial condition is invalid. Key must be 'IC Point' or 'Initial Storage Elev'. Received {key}"
            raise ValueError(msg)

    @classmethod
    def from_string(cls, in_string: str) -> Self:
        """Convert a string (row) to an initial condition."""
        splitted = in_string.split("=")
        ic_type = cls._parse_ic_type(splitted[0])
        v = "=".join(splitted[1:])
        table = v.replace(" ", "").split(",")
        idx = table[0]
        elevation = table[1]
        return cls(idx, elevation, ic_type)


class InitialConditions:
    """A list of initial conditions for a HEC-RAS unsteady flow file."""

    def __init__(self, ics: list[InitialCondition] | None = None, use_restart: str = "0"):
        """Class constructor."""
        self.ics = ics or []
        self.use_restart = use_restart
        self.line_triggers = {
            "Use Restart": self._set_use_restart,
            "IC Point Elev": self.append_ic,
            "Initial Storage Elev": self.append_ic,
        }

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        ic_txt = "\n".join([str(i) for i in self.ics])
        return f"Use Restart= {self.use_restart} \n{ic_txt}"

    def append_ic(self, ic: str) -> None:
        """Add an initial condition to the initial conditions block."""
        self.ics.append(InitialCondition.from_string(ic))

    def _set_use_restart(self, val: str) -> None:
        self.use_restart = val.split("=")[1].replace(" ", "")


class BoundaryCondition:
    """Parent class for all boundary conditions."""

    header_template: str = """
Boundary Location={pt_a},{pt_b},{pt_c},{pt_d},{sa_2d_id},{mesh_name},{pt_g},{bc_line_id},{pt_i}
"""
    body_template: str

    def __init__(self, sa_2d_id: str, mesh_name: str, bc_line_id: str):
        """Class constructor."""
        self.sa_2d_id = sa_2d_id
        self.mesh_name = mesh_name
        self.bc_line_id = bc_line_id

    def __str__(self) -> str:
        """Single line that will go in the text file."""
        if self.body is not None:
            return f"{self.header}{self.body}"
        else:
            return self.header

    @property
    def header(self) -> str:
        """Formatted header block for a boundary condition."""
        return self.header_template.format(
            pt_a="".ljust(16),
            pt_b="".ljust(16),
            pt_c="".ljust(8),
            pt_d="".ljust(8),
            pt_g="".ljust(16),
            pt_i="".ljust(32),
            sa_2d_id=self.sa_2d_id.ljust(16),
            mesh_name=self.mesh_name.ljust(16),
            bc_line_id=self.bc_line_id.ljust(32),
        )

    @property
    def body(self) -> str:
        """Formatted properties for the boundary condition."""
        return self.body_template.format(**self.__dict__)


class NormalDepthBoundaryCondition(BoundaryCondition):
    """A normal-depth boundary condition."""

    body_template: str = """
Friction Slope={friction_slope},{bc_params_2d}
"""

    def __init__(self, friction_slope: str, bc_params_2d: str = "0", **bc_kwargs: Any):
        """Class constructor."""
        super().__init__(**bc_kwargs)
        self.friction_slope = friction_slope
        self.bc_params_2d = bc_params_2d

    @classmethod
    def from_lines(cls, lines: str) -> Self:
        """Load boundary condition from text block."""
        for i in lines.split("\n"):
            if i.startswith("Boundary Location="):
                bc_kwargs = _parse_bc_header(i)
            if i.startswith("Friction"):
                data = i.split("=")[1].split(",")
                friction_slope = data[0]
                bc_params_2d = data[1]
        return cls(friction_slope, bc_params_2d, **bc_kwargs)


def _parse_bc_header(in_string: str) -> dict[str, str]:
    data = in_string.replace("Boundary Location=", "").split(",")
    sa_2d_id = data[4]
    mesh_name = data[5]
    bc_line_id = data[7]
    return {"sa_2d_id": sa_2d_id, "mesh_name": mesh_name, "bc_line_id": bc_line_id}


class FlowBoundaryCondition(BoundaryCondition):
    """A flow hydrograph boundary condition."""

    body_template: str = """
Interval={interval}
Flow Hydrograph={flow_hydrograph}
Stage Hydrograph TW Check=0
Flow Hydrograph Slope= {flow_hydrograph_slope}
DSS File={dss_file}
DSS Path={dss_path}
Use DSS=True
Use Fixed Start Time=False
Fixed Start Date/Time=,
Is Critical Boundary=False
Critical Boundary Flow=
"""  # TODO: When adding features, more wildcards will be needed

    def __init__(self, interval: str = "", dss_file: str = "", dss_path: str = "", **bc_kwargs: Any):
        """Class constructor."""
        super().__init__(**bc_kwargs)
        self.interval = interval
        self.dss_file = dss_file
        self.dss_path = dss_path
        self.flow_hydrograph = " 0 "
        self.flow_hydrograph_slope = ""

    @classmethod
    def from_lines(cls, lines: str) -> Self:
        """Load boundary condition parameters from a text block."""
        kwargs = {}
        for i in lines.split("\n"):
            if i.startswith("Boundary Location="):
                kwargs.update(_parse_bc_header(i))
            if i.startswith("Interval"):
                kwargs["interval"] = i.split("=")[1]
            if i.startswith("DSS File"):
                kwargs["dss_file"] = i.split("=")[1]
            if i.startswith("DSS Path"):
                kwargs["dss_path"] = i.split("=")[1]
        return cls(**kwargs)


def boundary_condition_factory(lines: str) -> BoundaryCondition:
    """Scan a text block and subclass BoundaryCondition appropriately."""
    for i in lines.split("\n"):
        if i.startswith("Flow Hydrograph="):
            return FlowBoundaryCondition.from_lines(lines)
        if i.startswith("Friction Slope="):
            return NormalDepthBoundaryCondition.from_lines(lines)
    raise RuntimeError(f"No boundary condition found in text block\n{lines}")


class BoundaryConditions:
    """All the boundary conditions for the flow file."""

    template: str = ""

    def __init__(self, bcs: list[BoundaryCondition] | None = None):
        """Class constructor."""
        self.bcs = bcs or []
        self.line_triggers = {"Boundary Location": self.append_bc}

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        return "\n".join([str(i) for i in self.bcs])

    def append_bc(self, lines: str) -> None:
        """Add a boundary condition to the file."""
        self.bcs.append(boundary_condition_factory(lines))


class MetBoundaryCondition:
    """Meteorological data boundary condition."""

    row_template: str = "Met BC={param}|{key}={val}"
    param: str = ""

    def __init__(
        self,
        expanded_view: str | None = "0",
        pt_interpolation: str | None = "Nearest",
        gridded_source: str = "DSS",
    ):
        """Class constructor."""
        self.expanded_view = expanded_view
        self.pt_interpolation = pt_interpolation
        self.gridded_source = gridded_source

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        return "\n".join(
            [
                self.row_template.format(param=self.param, key=k, val=v)
                for k, v in self.attributes.items()
                if v is not None
            ]
        )

    def _parse_attribute(self, val: str) -> None:
        key_val = val.split("|")[1].split("\n")[0]
        k, v = key_val.split("=")
        self.attributes[k] = v

    @property
    def line_triggers(self) -> dict[str, Callable[[str], None]]:
        """List of line starts that refer to this class."""
        return {f"Met BC={self.param}|{i}": self._parse_attribute for i in self.attributes}

    @property
    def attributes(self) -> dict[str, str | None | BC_PRECIP_MODE]:
        """Attributes that will be written to file."""
        return {
            "Expanded View": self.expanded_view,
            "Point Interpolation": self.pt_interpolation,
            "Gridded Source": self.gridded_source,
        }


class Precipitation(MetBoundaryCondition):
    """Precipitation data for the flow file."""

    param = "Precipitation"

    def __init__(
        self,
        mode: BC_PRECIP_MODE | None = None,
        expanded_view: str | None = "0",
        constant_value: str | None = None,
        constant_units: PRECIP_UNITS | None = None,
        pt_interpolation: PRECIP_INTERPOLATION_METHODS | None = "Nearest",
        gridded_source: str = "DSS",
        dss_filename: str | None = None,
        dss_filepath: str | None = None,
    ):
        """Class constructor."""
        self.mode = mode
        self.expanded_view = expanded_view
        self.constant_value = constant_value
        self.constant_units = constant_units
        self.pt_interpolation = pt_interpolation
        self.gridded_source = gridded_source
        self.dss_filename = dss_filename
        self.dss_filepath = dss_filepath

    @property
    def attributes(self) -> dict[str, str | None | BC_PRECIP_MODE]:
        """Attributes that will be written to file."""
        return {
            "Mode": self.mode,
            "Expanded View": self.expanded_view,
            "Constant Value": self.constant_value,
            "Constant Units": self.constant_units,
            "Point Interpolation": self.pt_interpolation,
            "Gridded Source": self.gridded_source,
            "Gridded DSS Filename": self.dss_filename,
            "Gridded DSS Pathname": self.dss_filepath,
        }


class Evapotranspiration(MetBoundaryCondition):
    """Evapotranspiration data for the flow file."""

    param = "Evapotranspiration"

    def __init__(
        self,
        mode: BC_ET_MODE | None = None,
        expanded_view: str | None = "0",
        constant_value: str | None = None,
        constant_units: ET_UNITS | None = None,
        pt_interpolation: ET_INTERPOLATION_METHODS | None = "Nearest",
        gridded_source: str = "DSS",
        dss_filename: str | None = None,
        dss_filepath: str | None = None,
    ):
        """Class constructor."""
        self.mode = mode
        self.expanded_view = expanded_view
        self.constant_value = constant_value
        self.constant_units = constant_units
        self.pt_interpolation = pt_interpolation
        self.gridded_source = gridded_source
        self.dss_filename = dss_filename
        self.dss_filepath = dss_filepath

    @property
    def attributes(self) -> dict[str, str | None | BC_PRECIP_MODE]:
        """Attributes that will be written to file."""
        return {
            "Mode": self.mode,
            "Expanded View": self.expanded_view,
            "Constant Value": self.constant_value,
            "Constant Units": self.constant_units,
            "Point Interpolation": self.pt_interpolation,
            "Gridded Source": self.gridded_source,
            "Gridded DSS Filename": self.dss_filename,
            "Gridded DSS Pathname": self.dss_filepath,
        }


class WindSpeed(MetBoundaryCondition):
    """Wind Speed data for the flow file."""

    param = "Wind Speed"


class WindDirection(MetBoundaryCondition):
    """Wind Direction data for the flow file."""

    param = "Wind Direction"


class WindVelocityX(MetBoundaryCondition):
    """Wind Velocity data for the flow file."""

    param = "Wind Velocity X"


class WindVelocityY(MetBoundaryCondition):
    """Wind Velocity data for the flow file."""

    param = "Wind Velocity Y"


class Humidity(MetBoundaryCondition):
    """Humidity data for the flow file."""

    param = "Humidity"


class AirTemperature(MetBoundaryCondition):
    """Air temperature data for the flow file."""

    param = "Air Temperature"


class AirDensity(MetBoundaryCondition):
    """Air Density data for the flow file."""

    param = "Air Density"

    def __init__(
        self,
        expanded_view: str | None = "0",
        constant_value: str | None = "1.225",
    ):
        """Class constructor."""
        self.mode = "Constant"
        self.expanded_view = expanded_view
        self.constant_value = constant_value
        self.constant_units = "kg/m3"
        self.pt_interpolation = "Nearest"
        self.gridded_source = "DSS"

    @property
    def attributes(self) -> dict[str, str | None | BC_PRECIP_MODE]:
        """Attributes that will be written to file."""
        return {
            "Mode": self.mode,
            "Expanded View": self.expanded_view,
            "Constant Value": self.constant_value,
            "Constant Units": self.constant_units,
            "Point Interpolation": self.pt_interpolation,
            "Gridded Source": self.gridded_source,
        }


class Wind:
    """Wind data for the flow file."""

    def __init__(
        self,
        mode: WIND_MODE = "No Wind Forces",
        wind_speed: WindSpeed | None = None,
        wind_direction: WindDirection | None = None,
        wind_velocity_x: WindVelocityX | None = None,
        wind_velocity_y: WindVelocityY | None = None,
        air_density: AirDensity | None = None,
    ):
        """Class constructor."""
        self.mode = mode
        self.wind_speed = wind_speed or WindSpeed()
        self.wind_direction = wind_direction or WindDirection()
        self.wind_velocity_x = wind_velocity_x or WindVelocityX()
        self.wind_velocity_y = wind_velocity_y or WindVelocityY()
        self.air_density = air_density or AirDensity()

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        atr_array = [
            str(i)
            for i in [
                self.wind_speed,
                self.wind_direction,
                self.wind_velocity_x,
                self.wind_velocity_y,
                self.air_density,
            ]
        ]
        return "\n".join(atr_array)

    @property
    def line_triggers(self) -> dict[str, Callable[[str], None]]:
        """List of line starts that refer to this class."""
        trigger_map = {}
        for _, v in self.__dict__.items():
            if hasattr(v, "line_triggers"):
                for k2, v2 in v.line_triggers.items():
                    trigger_map[k2] = v2
        trigger_map["Wind Mode"] = self._parse_wind_mode
        return trigger_map

    def _parse_wind_mode(self, val: str) -> None:
        self.mode = cast(WIND_MODE, val.split("=")[1])


class AirPressure(MetBoundaryCondition):
    """Air Pressure data for the flow file."""

    param = "Air Pressure"

    def __init__(
        self,
        constant_value: str | None = "1013.2",
    ):
        """Class constructor."""
        self.mode = "Constant"
        self.expanded_view = "0"
        self.constant_value = constant_value
        self.constant_units = "mb"
        self.pt_interpolation = "Inv Distance"
        self.gridded_source = "DSS"

    @property
    def attributes(self) -> dict[str, str | None | BC_PRECIP_MODE]:
        """Attributes that will be written to file."""
        return {
            "Mode": self.mode,
            "Expanded View": self.expanded_view,
            "Constant Value": self.constant_value,
            "Constant Units": self.constant_units,
            "Point Interpolation": self.pt_interpolation,
            "Gridded Source": self.gridded_source,
        }


class FlowOptions:
    """Miscellaneous options for the flow file (found in drop-down menu)."""

    template: str = """Non-Newtonian Method= 0 ,
Non-Newtonian Constant Vol Conc=0
Non-Newtonian Yield Method= 0 ,
Non-Newtonian Yield Coef=0, 0
User Yeild=   0
Non-Newtonian Sed Visc= 0 ,
Non-Newtonian Obrian B=0
User Viscosity=0
User Viscosity Ratio=0
Herschel-Bulkley Coef=0, 0
Clastic Method= 0 ,
Coulomb Phi=0
Voellmy X=0
Non-Newtonian Hindered FV= 0
Non-Newtonian FV K=0
Non-Newtonian ds=0
Non-Newtonian Max Cv=0
Non-Newtonian Bulking Method= 0 ,
Non-Newtonian High C Transport= 0 ,
"""

    def __init__(self) -> None:
        """Class constructor."""
        return  # TODO: Allow customization

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        return self.template


class UnsteadyFlowFile:
    """OOP representation of a HEC-RAS unsteady flow file."""

    template: str = """
Flow Title={flow_title}
Program Version={program_version}
BEGIN FILE DESCRIPTION:
{file_description}
END FILE DESCRIPTION:
{initial_conditions}
{boundary_conditions}
Met Point Raster Parameters=,,,,
Precipitation Mode={precip_mode}
Wind Mode={wind_mode}
Air Density Mode={air_density_mode}
{precipitation}
{evapotranspiration}
{wind}
{air_temperature}
{humidity}
{air_pressure}
{options}
"""

    def __init__(
        self,
        flow_title: str,
        program_version: str = "6.6",
        file_description: str = "",
        initial_conditions: InitialConditions | None = None,
        boundary_conditions: BoundaryConditions | None = None,
        precip_mode: PRECIP_MODE | None = "Disable",
        precipitation: Precipitation | None = None,
        evapotranspiration: Evapotranspiration | None = None,
        wind: Wind | None = None,
        air_temperature: AirTemperature | None = None,
        humidity: Humidity | None = None,
        air_pressure: AirPressure | None = None,
        options: FlowOptions | None = None,
    ):
        """Class constructor."""
        self.flow_title = flow_title
        self.program_version = program_version
        self.file_description = file_description
        self.initial_conditions = initial_conditions or InitialConditions()
        self.boundary_conditions = boundary_conditions or BoundaryConditions()
        self.precip_mode = precip_mode
        self.precipitation = precipitation or Precipitation()
        self.evapotranspiration = evapotranspiration or Evapotranspiration()
        self.wind = wind or Wind()
        self.wind_mode = self.wind.mode
        self.air_density_mode = ""
        self.air_temperature = air_temperature or AirTemperature()
        self.humidity = humidity or Humidity()
        self.air_pressure = air_pressure or AirPressure()
        self.options = options or FlowOptions()

    def __str__(self) -> str:
        """Lines that will go in the text file."""
        return "\n".join(self.lines)

    @property
    def lines(self) -> list[str]:
        """The flow file split by line breaks."""
        ls = self.template.format(**self.__dict__).split("\n")
        while "" in ls:
            ls.remove("")
        return [i + "\n" for i in ls]

    @property
    def line_triggers(self) -> dict[str, Callable[[str], None]]:
        """List of line starts that refer to this class."""
        trigger_map = {}
        for _, v in self.__dict__.items():
            if hasattr(v, "line_triggers"):
                for k2, v2 in v.line_triggers.items():
                    trigger_map[k2] = v2
        trigger_map["Flow Title"] = self._parse_flow_title
        trigger_map["Program Version"] = self._parse_program_version
        trigger_map["BEGIN FILE DESCRIPTION:"] = self._parse_file_description
        trigger_map["Met Point Raster Parameters"] = None
        trigger_map["Precipitation Mode"] = self._parse_precip_mode
        trigger_map["Wind Mode"] = self._parse_wind_mode
        trigger_map["Air Density Mode"] = self._parse_air_density_mode
        return trigger_map

    @classmethod
    def from_string(cls, in_str: str) -> Self:
        """Import an unsteady flow file."""
        lines = in_str.split("\n")
        tmp_cls = cls("")
        triggers = tmp_cls.line_triggers
        i = 0
        buffer: list[str] = []
        consumer = None
        while i < len(lines):
            if "=" in lines[i]:
                splitted = lines[i].split("=")
                reformat = "=".join(splitted[:-1])  # TODO: This could be more elegant with regex
            else:
                reformat = lines[i]

            if reformat in triggers:
                # Handle previous block
                if consumer is not None:
                    consumer("\n".join(buffer))
                # Initiate new block
                consumer = triggers[reformat]
                buffer = []
            buffer.append(lines[i])
            i += 1
        if consumer is not None:
            consumer("\n".join(buffer))
        return tmp_cls

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Read a flow file from disk."""
        with open(path) as f:
            lines = f.read()
        return cls.from_string(lines)

    def _parse_program_version(self, val: str) -> None:
        self.program_version = val.split("=")[1]

    def _parse_flow_title(self, val: str) -> None:
        self.flow_title = val.split("=")[1]

    def _parse_precip_mode(self, val: str) -> None:
        self.precip_mode = cast(PRECIP_MODE, val.split("=")[1])

    def _parse_wind_mode(self, val: str) -> None:
        self.wind.mode = cast(WIND_MODE, val.split("=")[1])

    def _parse_air_density_mode(self, val: str) -> None:
        self.air_density_mode = val.split("=")[1]

    def _parse_file_description(self, val: str) -> None:
        lines = []
        for i in val.split("\n"):
            if (not i.startswith("BEGIN FILE DESCRIPTION:")) and (not i.startswith("END FILE DESCRIPTION:")):
                lines.append(i)
        self.file_description = "\n".join(lines)

    def to_file(self, path: str) -> None:
        """Write the flow file to disk."""
        with open(path, mode="w", encoding="ascii", newline="\r\n") as f:
            f.writelines(self.lines)
