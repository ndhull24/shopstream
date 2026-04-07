with orders as (
    select * from {{ ref('stg_orders') }}
),

items as (
    select * from {{ ref('stg_order_items') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

order_summary as (
    select
        o.order_id,
        o.customer_id,
        c.full_name                             as customer_name,
        c.segment                               as customer_segment,
        o.order_date,
        o.order_year,
        o.order_month,
        o.status,
        o.shipping_city,
        o.shipping_state,
        count(i.item_id)                        as num_items,
        round(sum(i.line_total), 2)             as order_total,
        round(avg(i.discount), 4)               as avg_discount,
        round(avg(p.margin), 4)                 as avg_margin
    from orders o
    left join customers c  on o.customer_id  = c.customer_id
    left join items i      on o.order_id     = i.order_id
    left join products p   on i.product_id   = p.product_id
    group by 1,2,3,4,5,6,7,8,9,10
)

select * from order_summary