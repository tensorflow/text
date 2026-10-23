<x-mail::message>
Thank You For Registering with us
Here Is Your Password : {{$password}}

<x-mail::button :url="config('app.url')">
Login
</x-mail::button>
Please Change Your Password After Login

Thanks,<br>
{{ config('app.name') }}
</x-mail::message>
