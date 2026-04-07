with source as (
    select * from raw_orders
),

cleaned as (
    select
        order_id,
        customer_id,
        cast(order_date as date)                        as order_date,
        status,
        order_total,
        coalesce(shipping_city, 'Unknown')              as shipping_city,
        coalesce(shipping_state, 'Unknown')             as shipping_state,
        date_part('year',  cast(order_date as date))    as order_year,
        date_part('month', cast(order_date as date))    as order_month,
        date_part('dow',   cast(order_date as date))    as order_dow
    from source
    where order_id is not null
)

select * from cleaned