<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class CountrySeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $countries = [
            [
                'name' => 'United Arab Emirates',
                'code' => 'UAE',

            ],
            [
                'name' => 'kuwait',
                'code' => 'KWT',

            ],
            [
                'name' => 'Kingdom of Saudi Arabia',
                'code' => 'KSA',


            ],
            [
                'name' => 'Bahrain',
                'code' => 'BHR',

            ],
            [
                'name' => 'Qatar',
                'code' => 'QAT',

            ],
        ];

        foreach ($countries as $countryData) {
            Country::create($countryData);
        }
    }
}