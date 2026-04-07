with source as (
    select * from raw_customers
),

deduplicated as (
    select *,
        row_number() over (
            partition by customer_id
            order by signup_date desc
        ) as rn
    from source
),

cleaned as (
    select
        customer_id,
        first_name,
        last_name,
        first_name || ' ' || last_name          as full_name,
        coalesce(email, 'unknown@shopstream.com') as email,
        city,
        state,
        country,
        segment,
        cast(signup_date as date)                as signup_date,
        coalesce(cast(age as integer), 0)        as age
    from deduplicated
    where rn = 1
      and customer_id is not null
)

select * from cleaned