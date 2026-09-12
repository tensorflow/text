<?php

namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Seeder;

class UserSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $users = [
            [
                'name' => 'Administrator',
                'email' => 'admin@gmail.com',
                'role' => 'admin',
                'password' => bcrypt('12345678'),
            ],
            [
                'name' => 'Customer',
                'email' => 'customer@gmail.com',
                'role' => 'customer',
                'password' => bcrypt('12345678'),
            ],
            [
                'name' => 'Super Admin',
                'email' => 'superAdmin@gmail.com',
                'role' => 'superAdmin',
                'password' => bcrypt('12345678'),
            ],
        ];

        foreach ($users as $userData) {
            User::create($userData);
        }
    }
}
