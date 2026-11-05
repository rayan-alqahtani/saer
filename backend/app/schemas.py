from pydantic import BaseModel, Field


class ListingCompareRequest(BaseModel):
    listing_ids: list[str] = Field(min_length=2, max_length=2)


class TabularPriceRequest(BaseModel):
    make: str
    model_std: str
    year: int = Field(ge=1950, le=2027)
    km_mileage: float = Field(ge=0)
    trim_std: str = ""
    city: str = ""
    transmission: str = ""
    type_fuel: str = ""
    condition: str = "used"


class AuthCredentials(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(default="", max_length=80)
